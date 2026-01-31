#!/usr/bin/env python3
"""
Compute implied vols using the Bachelier (normal) model from option chain CSVs.
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from option_utils import infer_forward_df, normalize_expiry, parse_date, price_from_row


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bachelier_price(option_type: str, forward: float, strike: float, dfactor: float, t: float, vol: float) -> float:
    """Return Bachelier (normal) option price."""
    if t <= 0.0:
        intrinsic = max(forward - strike, 0.0)
        if option_type == "p":
            intrinsic = max(strike - forward, 0.0)
        return dfactor * intrinsic
    if vol <= 0.0:
        intrinsic = max(forward - strike, 0.0)
        if option_type == "p":
            intrinsic = max(strike - forward, 0.0)
        return dfactor * intrinsic
    d = (forward - strike) / (vol * math.sqrt(t))
    call = dfactor * ((forward - strike) * _norm_cdf(d) + vol * math.sqrt(t) * _norm_pdf(d))
    if option_type == "c":
        return call
    return call - dfactor * (forward - strike)


def bachelier_implied_vol(option_type: str, price: float, forward: float, strike: float, dfactor: float, t: float) -> float:
    """Return Bachelier implied vol via bisection."""
    if t <= 0.0 or price < 0.0:
        return float("nan")
    intrinsic = dfactor * max(forward - strike, 0.0)
    if option_type == "p":
        intrinsic = dfactor * max(strike - forward, 0.0)
    if price <= intrinsic + 1.0e-12:
        return 0.0

    lo = 1.0e-8
    hi = 0.1
    for _ in range(60):
        if bachelier_price(option_type, forward, strike, dfactor, t, hi) >= price:
            break
        hi *= 2.0
        if hi > 1e6:
            return float("nan")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = bachelier_price(option_type, forward, strike, dfactor, t, mid)
        if val >= price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main(argv: list[str]) -> None:
    """CLI entrypoint."""
    if len(argv) < 2:
        print("usage: python xbachelier_implied_vols.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]")
        print("       [--fwd-range L:U] [--plot [file.png]] [--plot-atm [file.png]] [--both-sides]")
        sys.exit(1)

    infile = argv[1]
    as_of = None
    plot = "--plot" in argv
    plot_file = None
    plot_atm = "--plot-atm" in argv
    plot_atm_file = None
    otm_only = "--both-sides" not in argv
    fwd_range = None
    expiry = None
    expiry_list = None
    expiry_range = None

    if "--as-of" in argv:
        as_of = normalize_expiry(argv[argv.index("--as-of") + 1])
    if "--plot" in argv:
        pidx = argv.index("--plot")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_file = argv[pidx + 1]
    if "--plot-atm" in argv:
        pidx = argv.index("--plot-atm")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_atm_file = argv[pidx + 1]
    if "--fwd-range" in argv:
        spec = argv[argv.index("--fwd-range") + 1]
        if ":" not in spec:
            raise ValueError("fwd-range must be L:U")
        left, right = spec.split(":", 1)
        fwd_range = (float(left), float(right))
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

    df = pd.read_csv(infile)
    lower_cols = {c.lower(): c for c in df.columns}
    has_bid = "bid" in lower_cols
    has_ask = "ask" in lower_cols
    if has_bid and has_ask:
        if lower_cols["bid"] != "bid" or lower_cols["ask"] != "ask":
            df = df.rename(columns={lower_cols["bid"]: "bid", lower_cols["ask"]: "ask"})
    else:
        price_cols = [c for c in df.columns if "price" in c.lower()]
        if not price_cols:
            missing = []
            if not has_bid:
                missing.append("bid")
            if not has_ask:
                missing.append("ask")
            print(f"error: missing columns {', '.join(missing)} and no price column found", file=sys.stderr)
            sys.exit(1)
        price_col = None
        for c in price_cols:
            if c.lower() in ("lastprice", "price"):
                price_col = c
                break
        if price_col is None:
            price_col = price_cols[0]
        if price_col != "lastPrice":
            df = df.rename(columns={price_col: "lastPrice"})
    symbol = None
    if "contractSymbol" in df.columns:
        first = df["contractSymbol"].dropna().astype(str).head(1)
        if not first.empty and len(first.iloc[0]) >= 3:
            symbol = first.iloc[0][:3]
    df["expiration"] = df["expiration"].astype(str).apply(normalize_expiry)
    df["option_type"] = df["option_type"].astype(str).str.lower()
    df["price"] = df.apply(price_from_row, axis=1, args=(True,))
    df = df.dropna(subset=["price", "strike", "option_type"])

    t0 = parse_date(as_of) if as_of else date.today()
    plot_data = []
    atm_rows = []

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

        def _iv(row: pd.Series) -> float:
            opt = "c" if row["option_type"] == "call" else "p"
            return bachelier_implied_vol(opt, row["price"], forward, row["strike"], dfactor, t)

        work["iv"] = work.apply(_iv, axis=1)
        work = work[np.isfinite(work["iv"]) & (work["iv"] >= 0.0)]
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

        pivot = work.pivot_table(index="strike", columns="option_type", values=["price", "iv"], aggfunc="first")
        pivot = pivot.sort_index()
        out = pd.DataFrame(index=pivot.index)
        out["c_price"] = pivot["price"].get("call")
        out["p_price"] = pivot["price"].get("put")
        out["c_iv"] = pivot["iv"].get("call")
        out["p_iv"] = pivot["iv"].get("put")
        out["c_iv_minus_p_iv"] = out["c_iv"] - out["p_iv"]

        print(f"\nexpiry: {exp}  forward: {forward:.4f}  df: {dfactor:.6f}")
        print(out.reset_index().to_string(index=False))

        avg_iv = out[["c_iv", "p_iv"]].mean(axis=1, skipna=True)
        plot_data.append((exp, out.index.values.astype(float), avg_iv.values))

        if plot_atm:
            if avg_iv.notna().any():
                atm_idx = int(np.argmin(np.abs(out.index.values.astype(float) - forward)))
                atm_strike = float(out.index.values.astype(float)[atm_idx])
                atm_iv = float(avg_iv.iloc[atm_idx])
            else:
                atm_strike = float("nan")
                atm_iv = float("nan")
            atm_rows.append(
                {
                    "expiry": exp,
                    "forward": forward,
                    "df": dfactor,
                    "atm_strike": atm_strike,
                    "atm_iv": atm_iv,
                }
            )

    if plot and plot_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for exp, strikes, iv in plot_data:
            ax.plot(strikes, iv, label=exp)
        title_date = as_of if as_of else datetime.now().date().isoformat()
        title_sym = symbol if symbol is not None else "data"
        ax.set_title(f"Bachelier implied vol vs strike by expiry ({title_sym}, as of {title_date})")
        ax.set_xlabel("strike")
        ax.set_ylabel("normal implied vol")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if plot_file:
            fig.savefig(plot_file, dpi=150)
            print(f"\nWrote plot to {plot_file}")
        else:
            plt.show()

    if plot_atm and atm_rows:
        atm_df = pd.DataFrame(atm_rows)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pd.to_datetime(atm_df["expiry"]), atm_df["atm_iv"], marker="o")
        title_date = as_of if as_of else datetime.now().date().isoformat()
        title_sym = symbol if symbol is not None else "data"
        ax.set_title(f"Bachelier ATM implied vol vs expiry ({title_sym}, as of {title_date})")
        ax.set_xlabel("expiry")
        ax.set_ylabel("normal implied vol")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if plot_atm_file:
            fig.savefig(plot_atm_file, dpi=150)
            print(f"\nWrote ATM plot to {plot_atm_file}")
        else:
            plt.show()


if __name__ == "__main__":
    main(sys.argv)
