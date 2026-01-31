#!/usr/bin/env python3
"""
Generate synthetic European option prices from SSVI/eSSVI and optionally fit back.
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from black_scholes import bs_price, implied_vol
from option_utils import normalize_expiry, parse_date
from ssvi import essvi_total_variance, fit_essvi, fit_ssvi, ssvi_total_variance
from scipy.optimize import least_squares


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
    essvi: bool,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
    a: float,
    b: float,
    c: float,
) -> pd.DataFrame:
    """Return synthetic options for one expiry."""
    rate = _rate_from_df(dfactor, t)
    spot = forward * dfactor
    q = 0.0

    rows = []
    for k in strikes:
        k_log = math.log(k / forward)
        if essvi:
            w = essvi_total_variance(
                np.array([k_log]), np.array([t]), a, b, c, rho, eta, gamma
            )[0]
        else:
            w = ssvi_total_variance(np.array([k_log]), theta, rho, eta, gamma)[0]
        if w <= 0.0:
            continue
        iv = math.sqrt(w / t)
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
    essvi: bool,
    fit_space: str,
) -> dict:
    """Fit SSVI/eSSVI and print a summary."""
    t0 = parse_date(as_of) if as_of else date.today()
    all_k = []
    all_w = []
    all_t = []
    all_strikes = []
    all_prices = []
    all_opts = []
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
            k = np.log(strikes / forward)
            w = (ivs ** 2) * t
            mask = np.isfinite(k) & np.isfinite(w) & (w > 0.0)
            if not np.any(mask):
                continue
            all_k.append(k[mask])
            all_w.append(w[mask])
            all_t.append(np.full_like(k[mask], t, dtype=float))
        else:
            k = np.log(strikes / forward)
            all_k.append(k)
            all_t.append(np.full_like(k, t, dtype=float))
            all_strikes.append(strikes)
            all_prices.append(prices)
            all_opts.append(opt_types)

    if not all_k:
        return {}

    k_all = np.concatenate(all_k)
    t_all = np.concatenate(all_t)

    if fit_space == "vol":
        w_all = np.concatenate(all_w)
        if essvi:
            params = fit_essvi(k_all, t_all, w_all)
            print(
                f"eSSVI fit: a={params['a']:.6f} b={params['b']:.6f} c={params['c']:.6f} "
                f"rho={params['rho']:.6f} eta={params['eta']:.6f} gamma={params['gamma']:.6f} "
                f"rmse={params['rmse']:.6f}"
            )
        else:
            params = fit_ssvi(k_all, w_all)
            print(
                f"SSVI fit: theta={params['theta']:.6f} rho={params['rho']:.6f} "
                f"eta={params['eta']:.6f} gamma={params['gamma']:.6f} rmse={params['rmse']:.6f}"
            )
        return params

    strikes_all = np.concatenate(all_strikes)
    prices_all = np.concatenate(all_prices)
    opts_all = np.concatenate(all_opts)
    spot = forward * dfactor
    q = 0.0

    if essvi:
        def resid(params: np.ndarray) -> np.ndarray:
            a_fit, b_fit, c_fit, rho_fit, eta_fit, gamma_fit = params
            w_fit = essvi_total_variance(k_all, t_all, a_fit, b_fit, c_fit, rho_fit, eta_fit, gamma_fit)
            w_fit = np.maximum(w_fit, 0.0)
            iv_fit = np.sqrt(w_fit / t_all)
            model_prices = []
            for kk, iv, opt, tt in zip(strikes_all, iv_fit, opts_all, t_all):
                rate = _rate_from_df(dfactor, tt)
                opt_code = "c" if opt == "call" else "p"
                model_prices.append(bs_price(opt_code, spot, kk, rate, tt, iv, q))
            return np.asarray(model_prices) - prices_all

        x0 = np.array([0.02, 0.1, 0.5, -0.2, 0.5, 0.5], dtype=float)
        lower = np.array([1e-6, 0.0, 0.05, -0.999, 1e-6, 0.01], dtype=float)
        upper = np.array([10.0, 10.0, 5.0, 0.999, 10.0, 5.0], dtype=float)
        res = least_squares(resid, x0, bounds=(lower, upper), max_nfev=8000)
        a_fit, b_fit, c_fit, rho_fit, eta_fit, gamma_fit = res.x
        rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
        params = {
            "a": float(a_fit),
            "b": float(b_fit),
            "c": float(c_fit),
            "rho": float(rho_fit),
            "eta": float(eta_fit),
            "gamma": float(gamma_fit),
            "rmse": rmse,
        }
        print(
            f"eSSVI fit: a={params['a']:.6f} b={params['b']:.6f} c={params['c']:.6f} "
            f"rho={params['rho']:.6f} eta={params['eta']:.6f} gamma={params['gamma']:.6f} "
            f"rmse={params['rmse']:.6f}"
        )
        return params

    def resid(params: np.ndarray) -> np.ndarray:
        theta_fit, rho_fit, eta_fit, gamma_fit = params
        w_fit = ssvi_total_variance(k_all, theta_fit, rho_fit, eta_fit, gamma_fit)
        w_fit = np.maximum(w_fit, 0.0)
        iv_fit = np.sqrt(w_fit / t_all)
        model_prices = []
        for kk, iv, opt, tt in zip(strikes_all, iv_fit, opts_all, t_all):
            rate = _rate_from_df(dfactor, tt)
            opt_code = "c" if opt == "call" else "p"
            model_prices.append(bs_price(opt_code, spot, kk, rate, tt, iv, q))
        return np.asarray(model_prices) - prices_all

    x0 = np.array([0.04, -0.2, 0.5, 0.5], dtype=float)
    lower = np.array([1e-6, -0.999, 1e-6, 0.01], dtype=float)
    upper = np.array([10.0, 0.999, 10.0, 5.0], dtype=float)
    res = least_squares(resid, x0, bounds=(lower, upper), max_nfev=5000)
    theta_fit, rho_fit, eta_fit, gamma_fit = res.x
    rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
    params = {
        "theta": float(theta_fit),
        "rho": float(rho_fit),
        "eta": float(eta_fit),
        "gamma": float(gamma_fit),
        "rmse": rmse,
    }
    print(
        f"SSVI fit: theta={params['theta']:.6f} rho={params['rho']:.6f} "
        f"eta={params['eta']:.6f} gamma={params['gamma']:.6f} rmse={params['rmse']:.6f}"
    )
    return params


def main(argv: list[str]) -> None:
    """CLI entrypoint."""
    if len(argv) < 2:
        print("usage: python xsynthetic_ssvi.py --outfile out.csv [--fit] [--plot [file.png]]")
        print("       [--as-of YYYY-MM-DD] [--expiry YYYYMMDD ... | --tenor-days D ...]")
        print("       [--fwd F] [--df DF] [--rate R] [--fwd-range L:U] [--n-strikes N]")
        print("       [--theta TH] [--rho RHO] [--eta ETA] [--gamma G] [--essvi]")
        print("       [--a A] [--b B] [--c C] [--noise N ...] [--spread S] [--both-sides]")
        print("       [--fit-vol | --fit-price]")
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
    theta = 0.04
    rho = -0.2
    eta = 0.5
    gamma = 0.5
    a = 0.02
    b = 0.1
    c = 0.5
    essvi = "--essvi" in argv
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
    if "--theta" in argv:
        theta = float(argv[argv.index("--theta") + 1])
    if "--rho" in argv:
        rho = float(argv[argv.index("--rho") + 1])
    if "--eta" in argv:
        eta = float(argv[argv.index("--eta") + 1])
    if "--gamma" in argv:
        gamma = float(argv[argv.index("--gamma") + 1])
    if "--a" in argv:
        a = float(argv[argv.index("--a") + 1])
    if "--b" in argv:
        b = float(argv[argv.index("--b") + 1])
    if "--c" in argv:
        c = float(argv[argv.index("--c") + 1])
    if "--noise" in argv:
        vals = _parse_multi(argv, "--noise")
        noise_vals = [float(v) for v in vals] if vals else [0.0]
    if "--spread" in argv:
        spread = float(argv[argv.index("--spread") + 1])
    if "--symbol" in argv:
        symbol = argv[argv.index("--symbol") + 1]

    expiries = _parse_multi(argv, "--expiry")
    tenor_days = _parse_multi(argv, "--tenor-days")
    tenor_days = [int(d) for d in tenor_days] if tenor_days else [30, 60, 90]
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

    if essvi:
        print(
            f"True eSSVI params: a={a:.6f} b={b:.6f} c={c:.6f} rho={rho:.6f} "
            f"eta={eta:.6f} gamma={gamma:.6f}"
        )
    else:
        print(
            f"True SSVI params: theta={theta:.6f} rho={rho:.6f} eta={eta:.6f} gamma={gamma:.6f}"
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
                    exp, t, fwd, dfactor, strikes, opt_types, noise, spread, symbol,
                    essvi, theta, rho, eta, gamma, a, b, c
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
            print(f"\nFitting SSVI for noise={noise:.6f}")
            params = _fit_and_report(out_df, expiry_list, fwd, dfactor, as_of, essvi, fit_space)
            if params:
                fit_results[noise] = params

    if plot and do_fit and expiry_grids and fit_results:
        multi_exp = len(expiry_grids) > 1
        for exp, grid in expiry_grids.items():
            t = grid["t"]
            strikes = grid["strikes"]
            k = np.log(strikes / fwd)
            if essvi:
                w_true = essvi_total_variance(k, np.full_like(k, t), a, b, c, rho, eta, gamma)
            else:
                w_true = ssvi_total_variance(k, theta, rho, eta, gamma)
            iv_true = np.sqrt(np.maximum(w_true, 0.0) / t) * 100.0
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(strikes, iv_true, linestyle="--", color="black", label="true")
            for noise, params in fit_results.items():
                if essvi:
                    w_fit = essvi_total_variance(
                        k, np.full_like(k, t), params["a"], params["b"], params["c"],
                        params["rho"], params["eta"], params["gamma"]
                    )
                else:
                    w_fit = ssvi_total_variance(k, params["theta"], params["rho"], params["eta"], params["gamma"])
                iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / t) * 100.0
                ax.plot(strikes, iv_fit, label=f"fit noise={noise:.4f}")
            tenor_days = int(round(t * 365.25))
            ax.set_title(f"SSVI true vs fit ({symbol}, expiry {exp}, T={tenor_days}d)")
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
