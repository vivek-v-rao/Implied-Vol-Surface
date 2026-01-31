#!/usr/bin/env python3
"""
Generate synthetic European option prices from SABR and optionally fit back.
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from black_scholes import bs_price, implied_vol
from option_utils import normalize_expiry, parse_date
from sabr import fit_sabr, sabr_iv_hagan


def _parse_multi(argv: list[str], flag: str) -> list[str]:
    """Collect a list of values following a flag."""
    if flag not in argv:
        return []
    idx = argv.index(flag) + 1
    vals = []
    while idx < len(argv) and not argv[idx].startswith("--"):
        vals.append(argv[idx])
        idx += 1
    return vals


def _as_expiries(as_of: str | None, tenor_days: list[int], expiries: list[str]) -> list[str]:
    """Return expiry dates (YYYY-MM-DD), rolling weekends forward."""
    if expiries:
        return [normalize_expiry(e) for e in expiries]
    t0 = parse_date(as_of) if as_of else date.today()
    out = []
    for d in tenor_days:
        exp = t0 + timedelta(days=d)
        if exp.weekday() == 5:
            exp = exp + timedelta(days=2)
        elif exp.weekday() == 6:
            exp = exp + timedelta(days=1)
        out.append(exp.isoformat())
    return out


def _rate_from_df(dfactor: float, t: float) -> float:
    """Return continuous rate from a discount factor."""
    if t <= 0.0:
        return 0.0
    return -math.log(dfactor) / t


def _generate_for_expiry(
    exp: str,
    t: float,
    forward: float,
    dfactor: float,
    strikes: np.ndarray,
    opt_types: list[str],
    noise: float,
    spread: float,
    symbol: str,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> pd.DataFrame:
    """Return synthetic options for one expiry."""
    rate = _rate_from_df(dfactor, t)
    spot = forward * dfactor
    q = 0.0

    rows = []
    for k in strikes:
        iv = sabr_iv_hagan(forward, k, t, alpha, beta, rho, nu)
        if not np.isfinite(iv) or iv <= 0.0:
            continue
        for opt in opt_types:
            price = bs_price(opt, spot, k, rate, t, iv, q)
            if noise > 0.0:
                price = price + np.random.normal(0.0, noise)
            price = max(price, 0.0)
            bid = max(price - spread / 2.0, 0.0)
            ask = price + spread / 2.0
            rows.append(
                {
                    "contractSymbol": f"{symbol}{exp.replace('-', '')}{opt.upper()}{k:.2f}",
                    "expiration": exp,
                    "strike": float(k),
                    "option_type": "call" if opt == "c" else "put",
                    "bid": float(bid),
                    "ask": float(ask),
                    "lastPrice": float(price),
                    "volume": 1,
                    "openInterest": 1,
                }
            )
    return pd.DataFrame(rows)


def _fit_and_report(
    df: pd.DataFrame,
    expiries: list[str],
    forward: float,
    dfactor: float,
    as_of: str | None,
    beta: float,
    fit_beta: bool,
    fit_space: str,
) -> list[dict]:
    """Fit SABR per expiry and print a summary."""
    t0 = parse_date(as_of) if as_of else date.today()
    rows = []
    for exp in expiries:
        work = df[df["expiration"] == exp].copy()
        if work.empty:
            continue
        t1 = parse_date(exp)
        if t1 == t0:
            t0_eff = t0 - timedelta(days=1)
        else:
            t0_eff = t0
        t = (t1 - t0_eff).days / 365.25
        if t <= 0:
            continue
        rate = _rate_from_df(dfactor, t)
        spot = forward * dfactor
        q = 0.0
        work["price"] = (work["bid"] + work["ask"]) / 2.0
        strikes = work["strike"].values.astype(float)
        prices = work["price"].values.astype(float)
        opt_types = work["option_type"].values.astype(str)
        mask = np.isfinite(strikes) & np.isfinite(prices) & (prices >= 0.0)
        strikes = strikes[mask]
        prices = prices[mask]
        opt_types = opt_types[mask]
        if strikes.size == 0:
            continue

        if fit_space == "vol":
            def _iv(row: pd.Series) -> float:
                opt = "c" if row["option_type"] == "call" else "p"
                return implied_vol(opt, row["price"], spot, row["strike"], rate, t, q, model="european")

            work["iv"] = work.apply(_iv, axis=1)
            work = work[np.isfinite(work["iv"]) & (work["iv"] > 0.0)]
            if work.empty:
                continue
            strikes = work["strike"].values.astype(float)
            ivs = work["iv"].values.astype(float)
            params = fit_sabr(forward, strikes, t, ivs, beta=beta, fit_beta=fit_beta)
        else:
            idx = int(np.argmin(np.abs(strikes - forward)))
            atm_price = prices[idx] if prices.size else np.nan
            atm_iv = 0.2
            if np.isfinite(atm_price):
                try:
                    atm_iv = implied_vol("c", atm_price, spot, strikes[idx], rate, t, q, model="european")
                except Exception:
                    atm_iv = 0.2
            alpha0 = atm_iv * (forward ** (1.0 - beta))
            alpha_upper = max(10.0, alpha0 * 10.0)

            if fit_beta:
                x0 = np.array([alpha0, beta, -0.1, 0.5], dtype=float)
                lower = np.array([1e-8, 0.0, -0.999, 1e-6], dtype=float)
                upper = np.array([alpha_upper, 1.0, 0.999, 10.0], dtype=float)

                def resid(params: np.ndarray) -> np.ndarray:
                    a_fit, b_fit, r_fit, n_fit = params
                    model_prices = []
                    for kk, opt in zip(strikes, opt_types):
                        iv = sabr_iv_hagan(forward, kk, t, a_fit, b_fit, r_fit, n_fit)
                        opt_code = "c" if opt == "call" else "p"
                        model_prices.append(bs_price(opt_code, spot, kk, rate, t, iv, q))
                    return np.asarray(model_prices) - prices
            else:
                x0 = np.array([alpha0, -0.1, 0.5], dtype=float)
                lower = np.array([1e-8, -0.999, 1e-6], dtype=float)
                upper = np.array([alpha_upper, 0.999, 10.0], dtype=float)

                def resid(params: np.ndarray) -> np.ndarray:
                    a_fit, r_fit, n_fit = params
                    model_prices = []
                    for kk, opt in zip(strikes, opt_types):
                        iv = sabr_iv_hagan(forward, kk, t, a_fit, beta, r_fit, n_fit)
                        opt_code = "c" if opt == "call" else "p"
                        model_prices.append(bs_price(opt_code, spot, kk, rate, t, iv, q))
                    return np.asarray(model_prices) - prices

            res = least_squares(resid, x0, bounds=(lower, upper), max_nfev=5000)
            if fit_beta:
                alpha_fit, beta_fit, rho_fit, nu_fit = res.x
            else:
                alpha_fit, rho_fit, nu_fit = res.x
                beta_fit = beta
            rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
            params = {
                "alpha": float(alpha_fit),
                "beta": float(beta_fit),
                "rho": float(rho_fit),
                "nu": float(nu_fit),
                "rmse": rmse,
            }
        rows.append(
            {
                "expiry": exp,
                "alpha_fit": params["alpha"],
                "beta_fit": params["beta"],
                "rho_fit": params["rho"],
                "nu_fit": params["nu"],
                "rmse": params["rmse"],
            }
        )
    if rows:
        print("\nSABR fit summary:")
        print(pd.DataFrame(rows).to_string(index=False))
    return rows


def main(argv: list[str]) -> None:
    """CLI entrypoint."""
    if len(argv) < 2:
        print("usage: python xsynthetic_sabr.py --outfile out.csv [--fit] [--plot [file.png]]")
        print("       [--as-of YYYY-MM-DD] [--expiry YYYYMMDD ... | --tenor-days D ...]")
        print("       [--fwd F] [--df DF] [--rate R] [--fwd-range L:U] [--n-strikes N]")
        print("       [--alpha A] [--beta B] [--rho RHO] [--nu NU] [--fit-beta]")
        print("       [--fit-vol | --fit-price]")
        print("       [--noise N ...] [--spread S] [--both-sides]")
        sys.exit(1)

    as_of = None
    outfile = None
    do_fit = "--fit" in argv
    fit_space = "price"
    plot = "--plot" in argv
    plot_file = None
    fwd = 100.0
    dfactor = None
    rate = 0.0
    fwd_range = (0.8, 1.2)
    n_strikes = 41
    alpha = 0.2
    beta = 1.0
    rho = -0.3
    nu = 0.5
    fit_beta = "--fit-beta" in argv
    noise_vals = [0.0]
    spread = 0.02
    both_sides = "--both-sides" in argv
    symbol = "SYN"

    if "--fit-vol" in argv:
        fit_space = "vol"
    if "--fit-price" in argv:
        fit_space = "price"
    if "--as-of" in argv:
        as_of = normalize_expiry(argv[argv.index("--as-of") + 1])
    if "--outfile" in argv:
        outfile = argv[argv.index("--outfile") + 1]
    if "--plot" in argv:
        pidx = argv.index("--plot")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_file = argv[pidx + 1]
    if "--fwd" in argv:
        fwd = float(argv[argv.index("--fwd") + 1])
    if "--df" in argv:
        dfactor = float(argv[argv.index("--df") + 1])
    if "--rate" in argv:
        rate = float(argv[argv.index("--rate") + 1])
    if "--fwd-range" in argv:
        spec = argv[argv.index("--fwd-range") + 1]
        left, right = spec.split(":", 1)
        fwd_range = (float(left), float(right))
    if "--n-strikes" in argv:
        n_strikes = int(argv[argv.index("--n-strikes") + 1])
    if "--alpha" in argv:
        alpha = float(argv[argv.index("--alpha") + 1])
    if "--beta" in argv:
        beta = float(argv[argv.index("--beta") + 1])
    if "--rho" in argv:
        rho = float(argv[argv.index("--rho") + 1])
    if "--nu" in argv:
        nu = float(argv[argv.index("--nu") + 1])
    if "--noise" in argv:
        vals = _parse_multi(argv, "--noise")
        noise_vals = [float(v) for v in vals] if vals else [0.0]
    if "--spread" in argv:
        spread = float(argv[argv.index("--spread") + 1])
    if "--symbol" in argv:
        symbol = argv[argv.index("--symbol") + 1]

    expiries = _parse_multi(argv, "--expiry")
    tenor_days = _parse_multi(argv, "--tenor-days")
    tenor_days = [int(d) for d in tenor_days] if tenor_days else [30]
    expiry_list = _as_expiries(as_of, tenor_days, expiries)

    if dfactor is None:
        if rate != 0.0:
            dfactor = math.exp(-rate * (tenor_days[0] / 365.25))
        else:
            dfactor = 1.0

    opt_types = ["c", "p"] if both_sides else ["c"]
    if not outfile and not do_fit:
        print("error: provide --outfile and/or --fit", file=sys.stderr)
        sys.exit(1)

    print(
        f"True SABR params: alpha={alpha:.6f} beta={beta:.6f} rho={rho:.6f} nu={nu:.6f}"
    )

    expiry_grids = {}
    for exp in expiry_list:
        t1 = parse_date(exp)
        t0 = parse_date(as_of) if as_of else date.today()
        if t1 == t0:
            t0_eff = t0 - timedelta(days=1)
        else:
            t0_eff = t0
        t = (t1 - t0_eff).days / 365.25
        if t <= 0:
            continue
        k_min = fwd * fwd_range[0]
        k_max = fwd * fwd_range[1]
        expiry_grids[exp] = {
            "t": t,
            "strikes": np.linspace(k_min, k_max, n_strikes),
        }

    fit_results = {}
    for noise in noise_vals:
        frames = []
        for exp in expiry_list:
            grid = expiry_grids.get(exp)
            if grid is None:
                continue
            t = grid["t"]
            strikes = grid["strikes"]
            frames.append(
                _generate_for_expiry(
                    exp, t, fwd, dfactor, strikes, opt_types, noise, spread, symbol, alpha, beta, rho, nu
                )
            )
        if not frames:
            continue
        out_df = pd.concat(frames, ignore_index=True)

        if outfile:
            if len(noise_vals) > 1:
                base, ext = outfile.rsplit(".", 1) if "." in outfile else (outfile, "csv")
                out_path = f"{base}_noise{noise:.4f}.{ext}"
            else:
                out_path = outfile
            out_df.to_csv(out_path, index=False)
            print(f"Wrote {len(out_df)} rows to {out_path}")

        if do_fit:
            print(f"\nFitting SABR for noise={noise:.6f}")
            fit_rows = _fit_and_report(out_df, expiry_list, fwd, dfactor, as_of, beta, fit_beta, fit_space)
            fit_results[noise] = fit_rows

    if plot and do_fit and expiry_grids and fit_results:
        multi_exp = len(expiry_grids) > 1
        for exp, grid in expiry_grids.items():
            t = grid["t"]
            strikes = grid["strikes"]
            iv_true = np.array([sabr_iv_hagan(fwd, k, t, alpha, beta, rho, nu) for k in strikes]) * 100.0
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(strikes, iv_true, linestyle="--", color="black", label="true")
            for noise, rows in fit_results.items():
                row = next((r for r in rows if r["expiry"] == exp), None)
                if row is None:
                    continue
                iv_fit = np.array(
                    [sabr_iv_hagan(fwd, k, t, row["alpha_fit"], row["beta_fit"],
                                   row["rho_fit"], row["nu_fit"]) for k in strikes]
                ) * 100.0
                ax.plot(strikes, iv_fit, label=f"fit noise={noise:.4f}")
            tenor_days = int(round(t * 365.25))
            ax.set_title(f"SABR true vs fit ({symbol}, expiry {exp}, T={tenor_days}d)")
            ax.set_xlabel("strike")
            ax.set_ylabel("implied vol (pct)")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if plot_file and not multi_exp:
                fig.savefig(plot_file, dpi=150)
                print(f"Wrote plot to {plot_file}")
            elif plot_file and multi_exp:
                base, ext = plot_file.rsplit(".", 1) if "." in plot_file else (plot_file, "png")
                out_path = f"{base}_{exp}.{ext}"
                fig.savefig(out_path, dpi=150)
                print(f"Wrote plot to {out_path}")
            else:
                plt.show()


if __name__ == "__main__":
    main(sys.argv)
