#!/usr/bin/env python3
"""
Shared option-analytics utilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd


def parse_date(s: str) -> datetime.date:
    """Parse YYYY-MM-DD or YYYYMMDD into a date."""
    if s.isdigit() and len(s) == 8:
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s, "%Y-%m-%d").date()


def normalize_expiry(s: str) -> str:
    """Normalize a date string to YYYY-MM-DD."""
    return parse_date(s).isoformat()


def price_from_row(row: pd.Series, use_lastprice: bool = False) -> float:
    """Return mid price, with optional lastPrice fallback."""
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    if use_lastprice:
        lp = row.get("lastPrice", np.nan)
        if pd.notna(lp) and lp > 0:
            return float(lp)
    return np.nan


def robust_parity_fit(k: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = b*K + a with trimming."""
    if len(k) < 3:
        raise ValueError("insufficient strikes to fit parity")
    b, a = np.polyfit(k, y, 1)
    resid = y - (b * k + a)
    if len(k) >= 6:
        keep = np.argsort(np.abs(resid))[: max(3, int(0.8 * len(k)))]
        b, a = np.polyfit(k[keep], y[keep], 1)
    return b, a


def infer_forward_df(
    df: pd.DataFrame,
    parity_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Infer forward and discount factor via put-call parity."""
    pivot = df.pivot_table(
        index="strike",
        columns="option_type",
        values="price",
        aggfunc="mean",
    )
    if "call" not in pivot.columns or "put" not in pivot.columns:
        raise ValueError("need both call and put prices to infer forward/df")
    pivot = pivot.dropna(subset=["call", "put"], how="any")
    if pivot.empty:
        raise ValueError("no strikes with both call and put prices")
    k = pivot.index.values.astype(float)
    y = (pivot["call"] - pivot["put"]).values.astype(float)
    b, a = robust_parity_fit(k, y)  # y = b*K + a
    dfactor = -b
    if not np.isfinite(dfactor) or dfactor <= 0:
        raise ValueError("invalid discount factor inferred from parity")
    forward = a / dfactor
    if parity_range is not None and np.isfinite(forward):
        low, high = parity_range
        low_k = forward * low
        high_k = forward * high
        sel = (k >= low_k) & (k <= high_k)
        if np.any(sel) and sel.sum() >= 3:
            b, a = robust_parity_fit(k[sel], y[sel])
            dfactor = -b
            if not np.isfinite(dfactor) or dfactor <= 0:
                raise ValueError("invalid discount factor inferred from parity (filtered)")
            forward = a / dfactor
    return forward, dfactor
