#!/usr/bin/env python3
"""
Compute implied vols for all strikes for a tenor without explicit S, r, q.
Uses put-call parity to infer forward (F) and discount factor (DF).

usage:
  python ximplied_vols.py options.csv [--expiry YYYY-MM-DD] [--as-of YYYY-MM-DD] [--fwd-range L:U] [--plot] [--both-sides] [--plot-atm] [--american] [--american-method bjerksund|crr] [--tree-steps N] [--debug-filters] [--spot S]
  python ximplied_vols.py --symbol SYMBOL [--expiry YYYYMMDD...] [--exp-range N:M] [--outfile file.csv] [--plot] [--american] [--american-method bjerksund|crr] [--tree-steps N] [--debug-filters] [--spot S]
for example
python ximplied_vols.py SPX_all.csv --expiry 20260201:20260320 --fwd-range 0.95:1.05 --plot spy.png
python ximplied_vols.py --symbol "^SPX" --exp-range 0:30 --fwd-range 0.95:1.05 --plot
"""

import sys
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from black_scholes import implied_vol
from option_chain_fetch import fetch_option_chain, get_expirations, get_spot
from option_utils import infer_forward_df, normalize_expiry, parse_date, price_from_row


def _parse_date(s: str) -> date:
    return parse_date(s)


def _normalize_expiry(s: str) -> str:
    return normalize_expiry(s)


def _has_valid_bid_ask(row: pd.Series) -> bool:
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    return pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0


def _spread_ok(row: pd.Series, max_spread_frac: float, min_mid: float) -> bool:
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if not (pd.notna(bid) and pd.notna(ask)):
        return False
    mid = 0.5 * (bid + ask)
    if not np.isfinite(mid) or mid < min_mid:
        return False
    spread = ask - bid
    if not np.isfinite(spread) or spread < 0:
        return False
    return (spread / mid) <= max_spread_frac




def implied_vols_for_tenor(
    df: pd.DataFrame,
    expiry: str,
    as_of: str | None = None,
    fwd_range: tuple[float, float] | None = None,
    otm_only: bool = True,
    american: bool = False,
    tree_steps: int = 200,
    debug_filters: bool = False,
    spot_override: float | None = None,
    american_method: str = "bjerksund",
) -> tuple[pd.DataFrame, float, float, float | None, float | None, float | None]:
    """Return implied vols for all strikes for the given expiry."""
    work = df.copy()
    if "expiration" not in work.columns:
        work["expiration"] = expiry
    else:
        work["expiration"] = work["expiration"].astype(str).apply(_normalize_expiry)
    work = work[work["expiration"] == _normalize_expiry(expiry)]
    if work.empty:
        raise ValueError(f"no rows for expiry {expiry}")
    work = work.copy()
    if debug_filters:
        print(f"debug {expiry}: start rows={len(work)}", file=sys.stderr)
    work["price"] = work.apply(price_from_row, axis=1, args=(False,))
    if american and "lastPrice" in work.columns:
        lp = work["lastPrice"]
        work.loc[work["price"].isna() & lp.notna() & (lp > 0), "price"] = lp
    filtered = work
    if american:
        max_spread_frac = 1.0
        min_mid = 0.0
        min_volume = 0
        min_open_interest = 0
    else:
        max_spread_frac = 0.10
        min_mid = 0.5
        min_volume = 1
        min_open_interest = 1
    if "bid" in work.columns and "ask" in work.columns:
        filtered = work[work.apply(_has_valid_bid_ask, axis=1)]
        filtered = filtered[filtered.apply(_spread_ok, axis=1, args=(max_spread_frac, min_mid))]
    if debug_filters:
        print(f"debug {expiry}: after bid/ask+spread rows={len(filtered)}", file=sys.stderr)
    if "volume" in work.columns:
        filtered = filtered[filtered["volume"].fillna(0) >= min_volume]
    if debug_filters:
        print(f"debug {expiry}: after volume rows={len(filtered)}", file=sys.stderr)
    if "openInterest" in work.columns:
        filtered = filtered[filtered["openInterest"].fillna(0) >= min_open_interest]
    if debug_filters:
        print(f"debug {expiry}: after openInterest rows={len(filtered)}", file=sys.stderr)
    filtered = filtered.dropna(subset=["price", "strike", "option_type"])
    if debug_filters:
        print(f"debug {expiry}: after dropna rows={len(filtered)}", file=sys.stderr)
    if filtered.empty:
        # relax filters and allow lastPrice fallback
        if "lastPrice" in work.columns:
            lp = work["lastPrice"]
            work.loc[work["price"].isna() & lp.notna() & (lp > 0), "price"] = lp
        filtered = work.dropna(subset=["price", "strike", "option_type"])
        if debug_filters:
            print(f"debug {expiry}: relaxed fallback rows={len(filtered)}", file=sys.stderr)
    parity_band = fwd_range if fwd_range is not None else (0.98, 1.02)
    forward, dfactor = infer_forward_df(filtered, parity_range=parity_band)
    if american and spot_override is not None and np.isfinite(spot_override):
        forward = spot_override
    if debug_filters:
        if spot_override is not None:
            print(f"debug {expiry}: spot_override={spot_override:.4f} forward_used={forward:.4f}", file=sys.stderr)
    t0 = _parse_date(as_of) if as_of else date.today()
    t1 = _parse_date(expiry)
    if t1 == t0:
        t0 = t0 - timedelta(days=1)
    t = (t1 - t0).days / 365.25
    if t <= 0:
        raise ValueError("expiry must be after as-of date")
    rate = -np.log(dfactor) / t
    if not np.isfinite(rate) or abs(rate) > 0.2:
        print(f"warning: unusual implied rate {rate:.4f} for expiry {expiry}", file=sys.stderr)
    spot = spot_override if spot_override is not None else (dfactor * forward)
    q = 0.0

    def _iv(row: pd.Series) -> float:
        opt = "c" if row["option_type"] == "call" else "p"
        model = "american" if american else "european"
        return implied_vol(opt, row["price"], spot, row["strike"], rate, t, q,
                           model=model, tree_steps=tree_steps, american_method=american_method)

    work["implied_vol"] = work.apply(_iv, axis=1) * 100.0
    if debug_filters:
        print(f"debug {expiry}: after iv rows={len(work)}", file=sys.stderr)
    # drop implausible vols and local outliers
    work = work[np.isfinite(work["implied_vol"])]
    if debug_filters:
        if not work.empty:
            print(f"debug {expiry}: iv min={work['implied_vol'].min():.4f} max={work['implied_vol'].max():.4f}", file=sys.stderr)
        print(f"debug {expiry}: after iv finite rows={len(work)}", file=sys.stderr)
    if american:
        iv_min = 0.0
        iv_max = 500.0
        outlier_thresh = 50.0
    else:
        iv_min = 2.0
        iv_max = 200.0
        outlier_thresh = 5.0
    work = work[(work["implied_vol"] >= iv_min) & (work["implied_vol"] <= iv_max)]
    if debug_filters:
        print(f"debug {expiry}: after iv bounds rows={len(work)}", file=sys.stderr)
    def _filter_outliers(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("strike")
        med = g["implied_vol"].rolling(window=5, center=True, min_periods=3).median()
        diff = (g["implied_vol"] - med).abs()
        return g[(diff.isna()) | (diff <= outlier_thresh)]
    if outlier_thresh is not None:
        filtered_groups = []
        for _, grp in work.groupby("option_type", sort=False):
            filtered_groups.append(_filter_outliers(grp))
        work = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else work
        if debug_filters:
            print(f"debug {expiry}: after outlier filter rows={len(work)}", file=sys.stderr)
    if work.empty:
        empty = pd.DataFrame(columns=["strike", "c_price", "c_iv", "p_price", "p_iv", "c_iv_minus_p_iv"])
        return empty, forward, dfactor, None, None, None
    out = work[["strike", "option_type", "price", "implied_vol"]].copy()
    out = out.sort_values(["strike", "option_type"])
    pivot = out.pivot_table(
        index="strike",
        columns="option_type",
        values=["price", "implied_vol"],
        aggfunc="mean",
    )
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns.to_flat_index()]
    pivot = pivot.reset_index()
    atm_strike = None
    atm_c_iv = None
    atm_p_iv = None
    if not pivot.empty and np.isfinite(forward):
        atm_ref = spot_override if (spot_override is not None and np.isfinite(spot_override)) else forward
        idx = (pivot["strike"] - atm_ref).abs().idxmin()
        atm_strike = float(pivot.loc[idx, "strike"])
        atm_c_iv = pivot.loc[idx, "implied_vol_call"] if "implied_vol_call" in pivot.columns else None
        atm_p_iv = pivot.loc[idx, "implied_vol_put"] if "implied_vol_put" in pivot.columns else None
    if fwd_range is not None:
        low, high = fwd_range
        fwd_for_range = forward
        min_k = pivot["strike"].min() if not pivot.empty else None
        max_k = pivot["strike"].max() if not pivot.empty else None
        if debug_filters:
            print(
                f"debug {expiry}: strike range=({min_k},{max_k}) fwd_range=({low},{high}) fwd_for_range={fwd_for_range}",
                file=sys.stderr,
            )
        if spot_override is not None and min_k is not None and max_k is not None:
            if spot_override < min_k or spot_override > max_k:
                print(
                    f"warning: spot {spot_override:.4f} outside strike range {min_k:.4f}-{max_k:.4f} for {expiry}",
                    file=sys.stderr,
                )
        if min_k is not None and max_k is not None and (min_k <= fwd_for_range <= max_k):
            low_k = fwd_for_range * low
            high_k = fwd_for_range * high
            pivot = pivot[(pivot["strike"] >= low_k) & (pivot["strike"] <= high_k)]
    rename_map = {
        "price_call": "c_price",
        "implied_vol_call": "c_iv",
        "price_put": "p_price",
        "implied_vol_put": "p_iv",
    }
    pivot = pivot.rename(columns=rename_map)
    cols = ["strike"]
    if "c_iv" in pivot.columns and "p_iv" in pivot.columns:
        pivot["c_iv_minus_p_iv"] = pivot["c_iv"] - pivot["p_iv"]
    for name in ("c_price", "c_iv", "p_price", "p_iv"):
        if name in pivot.columns:
            cols.append(name)
    if "c_iv_minus_p_iv" in pivot.columns:
        cols.append("c_iv_minus_p_iv")
    pivot = pivot[cols]
    return pivot, forward, dfactor, atm_strike, atm_c_iv, atm_p_iv


def _run_test() -> None:
    """Synthetic test using known S, r, q to verify recovered vols."""
    from black_scholes import bs_price

    spot = 100.0
    rate = 0.04
    q = 0.02
    vol = 0.25
    expiry = date.today() + timedelta(days=182)
    t = (expiry - date.today()).days / 365.25
    strikes = np.arange(80, 121, 5)
    rows = []
    for k in strikes:
        c = bs_price("c", spot, k, rate, t, vol, q)
        p = bs_price("p", spot, k, rate, t, vol, q)
        rows.append({"strike": k, "option_type": "call", "bid": c, "ask": c, "expiration": expiry.isoformat()})
        rows.append({"strike": k, "option_type": "put", "bid": p, "ask": p, "expiration": expiry.isoformat()})
    df = pd.DataFrame(rows)
    iv_df, _, _, _, _, _ = implied_vols_for_tenor(
        df, expiry.isoformat(), as_of=date.today().isoformat(), otm_only=False
    )
    if "c_iv" in iv_df.columns and "p_iv" in iv_df.columns:
        err_c = np.nanmax(np.abs(iv_df["c_iv"] - vol * 100.0))
        err_p = np.nanmax(np.abs(iv_df["p_iv"] - vol * 100.0))
        err = max(err_c, err_p)
        print(f"max abs iv error: {err:.6f}")
    print(iv_df.head(6).to_string(index=False))


def main(argv):
    if "--test" in argv:
        _run_test()
        return
    if len(argv) < 2:
        print("usage: python ximplied_vols.py options.csv [--expiry YYYY-MM-DD] [--as-of YYYY-MM-DD] [--fwd-range L:U] [--plot] [--both-sides] [--plot-atm]")
        sys.exit(1)

    infile = None
    expiry = None
    expiry_list = None
    expiry_range = None
    as_of = None
    fwd_range = None
    plot = "--plot" in argv
    plot_both = "--plot-both" in argv
    plot_atm = "--plot-atm" in argv
    otm_only = "--both-sides" not in argv
    american = "--american" in argv
    tree_steps = 200
    american_method = "bjerksund"
    debug_filters = "--debug-filters" in argv
    plot_file = None
    plot_atm_file = None
    symbol = None
    exp_range = None
    outfile = None
    spot_override = None
    if "--symbol" in argv:
        sidx = argv.index("--symbol")
        if sidx + 1 >= len(argv):
            raise ValueError("missing value for --symbol")
        symbol = argv[sidx + 1]
    else:
        infile = argv[1]
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
                expiry = _normalize_expiry(token)
        elif len(vals) > 1:
            expiry_list = [_normalize_expiry(v) for v in vals]
    if "--as-of" in argv:
        as_of = _normalize_expiry(argv[argv.index("--as-of") + 1])
    if "--fwd-range" in argv:
        spec = argv[argv.index("--fwd-range") + 1]
        if ":" not in spec:
            raise ValueError("fwd-range must be L:U")
        left, right = spec.split(":", 1)
        fwd_range = (float(left), float(right))
    if "--exp-range" in argv:
        exp_range = argv[argv.index("--exp-range") + 1]
    if "--outfile" in argv:
        outfile = argv[argv.index("--outfile") + 1]
    if "--tree-steps" in argv:
        tree_steps = int(argv[argv.index("--tree-steps") + 1])
    if "--american-method" in argv:
        american_method = argv[argv.index("--american-method") + 1].lower()
    if "--spot" in argv:
        spot_override = float(argv[argv.index("--spot") + 1])
    if "--plot" in argv:
        pidx = argv.index("--plot")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_file = argv[pidx + 1]
    if "--plot-atm" in argv:
        pidx = argv.index("--plot-atm")
        if pidx + 1 < len(argv) and not argv[pidx + 1].startswith("--"):
            plot_atm_file = argv[pidx + 1]

    if symbol is not None:
        # fetch fresh option chain
        if spot_override is None:
            spot_override = get_spot(symbol)
        if exp_range:
            df, _ = fetch_option_chain(symbol, None, exp_range)
        else:
            exps = None
            if expiry_list is not None:
                exps = expiry_list
            elif expiry is not None:
                exps = [expiry]
            elif expiry_range is not None:
                start_s, end_s = expiry_range.split(":", 1)
                start_d = _parse_date(start_s)
                end_d = _parse_date(end_s)
                if end_d < start_d:
                    start_d, end_d = end_d, start_d
                avail = get_expirations(symbol)
                exps = [e for e in avail if _parse_date(e) >= start_d and _parse_date(e) <= end_d]
            if exps is None:
                df, _ = fetch_option_chain(symbol, None, None)
            else:
                frames = []
                for exp in exps:
                    df_exp, _ = fetch_option_chain(symbol, exp, None)
                    frames.append(df_exp)
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if outfile:
            df.to_csv(outfile, index=False)
    else:
        df = pd.read_csv(infile)
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.lower()
    expiries = df["expiration"].dropna().unique() if "expiration" in df.columns else []
    expiries = [_normalize_expiry(str(e)) for e in expiries]
    if expiry_range is not None:
        start_s, end_s = expiry_range.split(":", 1)
        start_d = _parse_date(start_s)
        end_d = _parse_date(end_s)
        if end_d < start_d:
            start_d, end_d = end_d, start_d
        expiry_list = [
            e for e in expiries
            if _parse_date(e) >= start_d and _parse_date(e) <= end_d
        ]
    summary_rows = []
    large_diff_rows = []
    diff_thresh = 10.0
    plot_data = []
    if expiry_list is not None:
        for exp in expiry_list:
            out, forward, dfactor, atm_strike, atm_c_iv, atm_p_iv = implied_vols_for_tenor(
                df, exp, as_of=as_of, fwd_range=fwd_range, otm_only=otm_only,
                american=american, tree_steps=tree_steps, debug_filters=debug_filters,
                spot_override=spot_override, american_method=american_method
            )
            print(f"\nexpiry: {exp}  forward: {forward:.2f}  df: {dfactor:.4f}")
            out_print = out.copy()
            for col in ("c_iv", "p_iv", "c_iv_minus_p_iv"):
                if col in out_print.columns:
                    out_print[col] = out_print[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else x)
            print(out_print.to_string(index=False))
            summary_rows.append(
                {
                    "expiry": exp,
                    "forward": forward,
                    "df": dfactor,
                    "atm_strike": atm_strike,
                    "c_iv": atm_c_iv,
                    "p_iv": atm_p_iv,
                    "c_iv_minus_p_iv": (atm_c_iv - atm_p_iv) if (atm_c_iv is not None and atm_p_iv is not None) else None,
                }
            )
            plot_data.append((exp, out))
    elif expiry is None:
        if len(expiries) == 0:
            raise ValueError("no expiration column found; pass --expiry YYYY-MM-DD")
        for exp in sorted(expiries):
            out, forward, dfactor, atm_strike, atm_c_iv, atm_p_iv = implied_vols_for_tenor(
                df, exp, as_of=as_of, fwd_range=fwd_range, otm_only=otm_only,
                american=american, tree_steps=tree_steps, debug_filters=debug_filters,
                spot_override=spot_override, american_method=american_method
            )
            print(f"\nexpiry: {exp}  forward: {forward:.2f}  df: {dfactor:.4f}")
            out_print = out.copy()
            for col in ("c_iv", "p_iv", "c_iv_minus_p_iv"):
                if col in out_print.columns:
                    out_print[col] = out_print[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else x)
            print(out_print.to_string(index=False))
            summary_rows.append(
                {
                    "expiry": exp,
                    "forward": forward,
                    "df": dfactor,
                    "atm_strike": atm_strike,
                    "c_iv": atm_c_iv,
                    "p_iv": atm_p_iv,
                    "c_iv_minus_p_iv": (atm_c_iv - atm_p_iv) if (atm_c_iv is not None and atm_p_iv is not None) else None,
                }
            )
            if "c_iv" in out.columns and "p_iv" in out.columns:
                diff = (out["c_iv"] - out["p_iv"]).abs()
                mask = diff > diff_thresh
                if mask.any():
                    bad = out.loc[mask, ["strike", "c_iv", "p_iv", "c_iv_minus_p_iv"]].copy()
                    bad.insert(0, "expiry", exp)
                    bad["iv_diff"] = diff[mask].values
                    large_diff_rows.append(bad)
            plot_data.append((exp, out))
    else:
        out, forward, dfactor, atm_strike, atm_c_iv, atm_p_iv = implied_vols_for_tenor(
            df, expiry, as_of=as_of, fwd_range=fwd_range, otm_only=otm_only,
            american=american, tree_steps=tree_steps, debug_filters=debug_filters,
            spot_override=spot_override, american_method=american_method
        )
        print(f"expiry: {expiry}  forward: {forward:.2f}  df: {dfactor:.4f}")
        out_print = out.copy()
        for col in ("c_iv", "p_iv", "c_iv_minus_p_iv"):
            if col in out_print.columns:
                out_print[col] = out_print[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else x)
        print(out_print.to_string(index=False))
        summary_rows.append(
            {
                "expiry": expiry,
                "forward": forward,
                "df": dfactor,
                "atm_strike": atm_strike,
                "c_iv": atm_c_iv,
                "p_iv": atm_p_iv,
                "c_iv_minus_p_iv": (atm_c_iv - atm_p_iv) if (atm_c_iv is not None and atm_p_iv is not None) else None,
            }
        )
        if "c_iv" in out.columns and "p_iv" in out.columns:
            diff = (out["c_iv"] - out["p_iv"]).abs()
            mask = diff > diff_thresh
            if mask.any():
                bad = out.loc[mask, ["strike", "c_iv", "p_iv"]].copy()
                bad.insert(0, "expiry", expiry)
                bad["iv_diff"] = diff[mask].values
                large_diff_rows.append(bad)
        plot_data.append((expiry, out))
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\nATM vol summary:")
        summary_print = summary_df.copy()
        if "forward" in summary_print.columns:
            summary_print["forward"] = summary_print["forward"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else x
            )
        if "df" in summary_print.columns:
            summary_print["df"] = summary_print["df"].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else x
            )
        if "atm_strike" in summary_print.columns:
            non_na = summary_print["atm_strike"].dropna()
            if not non_na.empty and (non_na % 1 == 0).all():
                summary_print["atm_strike"] = summary_print["atm_strike"].astype(int)
        for col in ("c_iv", "p_iv", "c_iv_minus_p_iv"):
            if col in summary_print.columns:
                summary_print[col] = summary_print[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else x)
        print(summary_print.to_string(index=False))
    if large_diff_rows:
        bad_df = pd.concat(large_diff_rows, ignore_index=True)
        bad_df = bad_df.sort_values(["expiry", "iv_diff"], ascending=[True, False])
        bad_print = bad_df.copy()
        for col in ("c_iv", "p_iv", "c_iv_minus_p_iv", "iv_diff"):
            if col in bad_print.columns:
                bad_print[col] = bad_print[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else x)
        print(f"\nLarge call/put IV diffs (>{diff_thresh:.2f}):")
        print(bad_print.to_string(index=False))
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for --plot (pip install matplotlib).", file=sys.stderr)
            return
        if not plot_data:
            print("no data to plot", file=sys.stderr)
            return
        fig, ax = plt.subplots()
        for exp, out in plot_data:
            plot_out = out.copy()
            if "c_iv" in plot_out.columns and "p_iv" in plot_out.columns:
                iv_diff = (plot_out["c_iv"] - plot_out["p_iv"]).abs()
                plot_out = plot_out[(iv_diff <= diff_thresh) | iv_diff.isna()]
                if otm_only:
                    avg_iv = plot_out["c_iv"].where(plot_out["strike"] >= forward, plot_out["p_iv"])
                else:
                    avg_iv = plot_out[["c_iv", "p_iv"]].mean(axis=1, skipna=True)
                if avg_iv.notna().any():
                    ax.plot(plot_out["strike"], avg_iv, linestyle="-", label=f"{exp} avg")
                if plot_both:
                    if plot_out["c_iv"].notna().any():
                        ax.plot(plot_out["strike"], plot_out["c_iv"], linestyle="--", label=f"{exp} c")
                    if plot_out["p_iv"].notna().any():
                        ax.plot(plot_out["strike"], plot_out["p_iv"], linestyle=":", label=f"{exp} p")
            else:
                # fallback if only one side is present
                if "c_iv" in plot_out.columns and plot_out["c_iv"].notna().any():
                    ax.plot(plot_out["strike"], plot_out["c_iv"], linestyle="-", label=f"{exp} c")
                if "p_iv" in plot_out.columns and plot_out["p_iv"].notna().any():
                    ax.plot(plot_out["strike"], plot_out["p_iv"], linestyle="-", label=f"{exp} p")
        ax.set_xlabel("strike")
        ax.set_ylabel("implied vol")
        title_sym = symbol if symbol is not None else "data"
        title_date = _normalize_expiry(as_of) if as_of else datetime.now().date().isoformat()
        ax.set_title(f"Implied vol vs strike by expiry ({title_sym}, {title_date})")
        ax.grid(True, alpha=0.25)
        if ax.lines:
            ax.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        if plot_file:
            plt.savefig(plot_file, dpi=150)
        else:
            plt.show()
    if plot_atm and summary_rows:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for --plot-atm (pip install matplotlib).", file=sys.stderr)
            return
        atm_df = pd.DataFrame(summary_rows)
        atm_df = atm_df.dropna(subset=["expiry"])
        if atm_df.empty:
            print("no ATM data to plot", file=sys.stderr)
            return
        as_of_date = _parse_date(as_of) if as_of else date.today()
        atm_df["tenor_days"] = atm_df["expiry"].map(lambda x: (_parse_date(x) - as_of_date).days)
        atm_df = atm_df[atm_df["tenor_days"] > 0]
        if atm_df.empty:
            print("no positive tenors to plot", file=sys.stderr)
            return
        atm_df = atm_df.sort_values("tenor_days")
        atm_df["atm_avg_iv"] = atm_df[["c_iv", "p_iv"]].mean(axis=1, skipna=True)
        fig, ax = plt.subplots()
        ax.plot(atm_df["tenor_days"], atm_df["atm_avg_iv"], marker="o", label="ATM avg")
        if "c_iv" in atm_df.columns and atm_df["c_iv"].notna().any():
            ax.plot(atm_df["tenor_days"], atm_df["c_iv"], marker="o", linestyle="--", label="ATM call")
        if "p_iv" in atm_df.columns and atm_df["p_iv"].notna().any():
            ax.plot(atm_df["tenor_days"], atm_df["p_iv"], marker="o", linestyle=":", label="ATM put")
        ax.set_xlabel("tenor (days)")
        ax.set_ylabel("implied vol")
        title_sym = symbol if symbol is not None else "data"
        title_date = _normalize_expiry(as_of) if as_of else datetime.now().date().isoformat()
        ax.set_title(f"ATM implied vol vs tenor ({title_sym}, {title_date})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
        plt.tight_layout()
        if plot_atm_file:
            plt.savefig(plot_atm_file, dpi=150)
        else:
            plt.show()


if __name__ == "__main__":
    main(sys.argv)
