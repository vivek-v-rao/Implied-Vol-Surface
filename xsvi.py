#!/usr/bin/env python3
"""
Fit SVI smile per expiration from European option prices in a CSV file.

Expected columns (minimum):
  expiration, strike, option_type, bid, ask
Optional: lastPrice

Usage:
  python xsvi.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]
      [--fwd-range L:U] [--both-sides] [--outfile out.csv] [--plot [file.png]] [--plot-density [file.png]]
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from black_scholes import implied_vol
from option_utils import infer_forward_df, normalize_expiry, parse_date, price_from_row
from svi import fit_svi, svi_total_variance


def main(argv: list[str]) -> None:
    """CLI entrypoint."""
    if len(argv) < 2:
        print("usage: python xsvi.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]")
        print("       [--fwd-range L:U] [--both-sides] [--outfile out.csv] [--plot [file.png]] [--plot-density [file.png]]")
        sys.exit(1)

    infile = argv[1]
    as_of = None
    outfile = None
    plot = "--plot" in argv
    plot_file = None
    plot_density = "--plot-density" in argv
    plot_density_log_s = "--plot-density-log-s" in argv
    if plot_density_log_s:
        plot_density = True
    plot_density_file = None
    otm_only = "--both-sides" not in argv
    expiry = None
    expiry_list = None
    expiry_range = None
    fwd_range = None
    if "--as-of" in argv:
        as_of = normalize_expiry(argv[argv.index("--as-of") + 1])
    if "--outfile" in argv:
        outfile = argv[argv.index("--outfile") + 1]
    if "--plot" in argv:
        pidx = argv.index("--plot")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_file = argv[pidx + 1]
    if "--plot-density" in argv:
        pidx = argv.index("--plot-density")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_density_file = argv[pidx + 1]
    if "--plot-density-log-s" in argv:
        pidx = argv.index("--plot-density-log-s")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_density_file = argv[pidx + 1]
    if "--expiry" in argv:
        idx = argv.index("--expiry") + 1
        vals = []
        while idx < len(argv) and not argv[idx].startswith("--"):
            vals.append(argv[idx])
            idx += 1
        if len(vals) == 1:
            token = vals[0]
            if ":" in token:
                expiry_range = token
            else:
                expiry = normalize_expiry(token)
        elif len(vals) > 1:
            expiry_list = [normalize_expiry(v) for v in vals]
    if "--fwd-range" in argv:
        spec = argv[argv.index("--fwd-range") + 1]
        if ":" not in spec:
            raise ValueError("fwd-range must be L:U")
        left, right = spec.split(":", 1)
        fwd_range = (float(left), float(right))

    df = pd.read_csv(infile)
    df["expiration"] = df["expiration"].astype(str).apply(normalize_expiry)
    df["option_type"] = df["option_type"].astype(str).str.lower()
    df["price"] = df.apply(price_from_row, axis=1, args=(True,))
    df = df.dropna(subset=["price", "strike", "option_type"])

    t0 = parse_date(as_of) if as_of else date.today()
    results = []
    fit_rows = []
    plot_data = []
    density_data = []
    expiries = sorted(df["expiration"].unique())
    if expiry_range is not None:
        start_s, end_s = expiry_range.split(":", 1)
        start_d = parse_date(start_s)
        end_d = parse_date(end_s)
        if end_d < start_d:
            start_d, end_d = end_d, start_d
        expiries = [e for e in expiries if parse_date(e) >= start_d and parse_date(e) <= end_d]
    if expiry is not None:
        expiries = [expiry] if expiry in expiries else []
    if expiry_list is not None:
        expiries = [e for e in expiries if e in set(expiry_list)]

    for exp in expiries:
        work = df[df["expiration"] == exp].copy()
        if work.empty:
            continue
        try:
            forward, dfactor = infer_forward_df(work)
        except Exception as e:
            print(f"{exp}: parity inference failed ({e})", file=sys.stderr)
            continue

        t1 = parse_date(exp)
        if t1 == t0:
            t0_eff = t0 - timedelta(days=1)
        else:
            t0_eff = t0
        t = (t1 - t0_eff).days / 365.25
        if t <= 0:
            print(f"{exp}: non-positive time to expiry", file=sys.stderr)
            continue
        rate = -np.log(dfactor) / t
        spot = dfactor * forward
        q = 0.0

        def _iv(row: pd.Series) -> float:
            opt = "c" if row["option_type"] == "call" else "p"
            return implied_vol(opt, row["price"], spot, row["strike"], rate, t, q, model="european")

        work["iv"] = work.apply(_iv, axis=1)
        work = work[np.isfinite(work["iv"]) & (work["iv"] > 0.01) & (work["iv"] < 5.0)]
        if work.empty:
            print(f"{exp}: no valid implied vols", file=sys.stderr)
            continue

        if otm_only:
            work = work[
                ((work["option_type"] == "call") & (work["strike"] >= forward)) |
                ((work["option_type"] == "put") & (work["strike"] <= forward))
            ]
        if work.empty:
            print(f"{exp}: no OTM implied vols", file=sys.stderr)
            continue
        if fwd_range is not None:
            low, high = fwd_range
            low_k = forward * low
            high_k = forward * high
            work = work[(work["strike"] >= low_k) & (work["strike"] <= high_k)]
            if work.empty:
                print(f"{exp}: no strikes in fwd-range", file=sys.stderr)
                continue

        k = np.log(work["strike"].values.astype(float) / forward)
        w = (work["iv"].values ** 2) * t

        params = fit_svi(k, w)
        results.append(
            {
                "expiry": exp,
                "forward": forward,
                "df": dfactor,
                "rmse": params["rmse"],
                "a": params["a"],
                "b": params["b"],
                "rho": params["rho"],
                "m": params["m"],
                "sigma": params["sigma"],
            }
        )
        print(
            f"{exp}: a={params['a']:.6f} b={params['b']:.6f} rho={params['rho']:.6f} "
            f"m={params['m']:.6f} sigma={params['sigma']:.6f} rmse={params['rmse']:.6f}"
        )

        strikes_sorted = np.array(sorted(work["strike"].unique()), dtype=float)
        k_grid = np.log(strikes_sorted / forward)
        w_fit = svi_total_variance(k_grid, params["a"], params["b"], params["rho"],
                                   params["m"], params["sigma"])
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / t)
        for strike, iv in zip(strikes_sorted, iv_fit):
            fit_rows.append(
                {
                    "expiry": exp,
                    "strike": strike,
                    "iv_fit": float(iv),
                    "iv_fit_pct": float(iv * 100.0),
                }
            )
        plot_data.append((exp, strikes_sorted, iv_fit * 100.0))
        density_data.append((exp, strikes_sorted, iv_fit, forward, dfactor, t))

    if results:
        print("\nSVI summary:")
        print(pd.DataFrame(results).to_string(index=False))

    if outfile and fit_rows:
        out_df = pd.DataFrame(fit_rows)
        out_df.to_csv(outfile, index=False)
        print(f"\nWrote fitted vols to {outfile}")

    if plot and plot_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for exp, strikes, iv_pct in plot_data:
            ax.plot(strikes, iv_pct, label=exp)
        title_date = as_of if as_of else datetime.now().date().isoformat()
        ax.set_title(f"SVI implied vol vs strike by expiry (data, {title_date})")
        ax.set_xlabel("strike")
        ax.set_ylabel("implied vol (pct)")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if plot_file:
            fig.savefig(plot_file, dpi=150)
            print(f"\nWrote plot to {plot_file}")
        else:
            plt.show()

    if plot_density and density_data:
        from black_scholes import bs_price
        fig, ax = plt.subplots(figsize=(10, 6))
        single_tenor = len(density_data) == 1
        ref_x = None
        ref_density = None
        for exp, strikes, ivs, forward, dfactor, t in density_data:
            strikes = np.asarray(strikes, dtype=float)
            if strikes.size < 5:
                continue
            # Use a uniform strike grid to avoid zero spacing warnings
            if fwd_range is not None:
                k_min = forward * fwd_range[0]
                k_max = forward * fwd_range[1]
            else:
                # Default density window to avoid extreme tails
                k_min = forward * 0.8
                k_max = forward * 1.2
                k_min = max(k_min, float(np.nanmin(strikes)))
                k_max = min(k_max, float(np.nanmax(strikes)))
            if not np.isfinite(k_min) or not np.isfinite(k_max) or k_max <= k_min:
                continue
            grid = np.linspace(k_min, k_max, 200)
            k_grid = np.log(grid / forward)
            w_grid = svi_total_variance(k_grid, params["a"], params["b"], params["rho"],
                                        params["m"], params["sigma"])
            iv_grid = np.sqrt(np.maximum(w_grid, 0.0) / t)
            rate = -np.log(dfactor) / t
            q = 0.0
            spot = forward * dfactor
            call_prices = np.array([bs_price("c", spot, k, rate, t, v, q)
                                    for k, v in zip(grid, iv_grid)])
            dK = grid[1] - grid[0]
            dC_dK = np.gradient(call_prices, dK)
            d2C_dK2 = np.gradient(dC_dK, dK)
            # Breeden-Litzenberger: f_S(K) = exp(rT) * d2C/dK2
            f_s = np.maximum(d2C_dK2 * np.exp(rate * t), 0.0)
            if plot_density_log_s:
                x = np.log(grid)
                density = f_s * grid
                ax.plot(x, density, label=exp)
                if single_tenor:
                    ref_x = x
                    ref_density = density
            else:
                ax.plot(grid, f_s, label=exp)
        if plot_density_log_s and single_tenor and ref_x is not None and ref_density is not None:
            integral = np.trapezoid(ref_density, ref_x)
            if integral > 0:
                dens = ref_density / integral
                mu = float(np.trapezoid(ref_x * dens, ref_x))
                var = float(np.trapezoid((ref_x - mu) ** 2 * dens, ref_x))
                if var > 0:
                    sd = np.sqrt(var)
                    normal = (1.0 / (sd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((ref_x - mu) / sd) ** 2)
                    ax.plot(ref_x, normal, linestyle="--", color="black", label="normal ref")
        title_date = as_of if as_of else datetime.now().date().isoformat()
        if plot_density_log_s:
            ax.set_title(f"Implied density vs log stock (SVI, {title_date})")
            ax.set_xlabel("log(stock)")
        else:
            ax.set_title(f"Implied density vs stock (SVI, {title_date})")
            ax.set_xlabel("stock")
        ax.set_ylabel("density (arb units)")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if plot_density_file:
            fig.savefig(plot_density_file, dpi=150)
            print(f"\nWrote density plot to {plot_density_file}")
        else:
            plt.show()


if __name__ == "__main__":
    main(sys.argv)
