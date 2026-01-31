"""CEV utilities for European option pricing and fitting."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import ncx2


def cev_price(
    option_type: str,
    spot: float,
    strike: float,
    rate: float,
    time: float,
    sigma: float,
    beta: float,
    q: float = 0.0,
) -> float:
    """Return CEV European option price."""
    opt = option_type.lower()
    if opt.startswith("c"):
        opt = "c"
    elif opt.startswith("p"):
        opt = "p"
    else:
        return float("nan")
    if time <= 0.0:
        intrinsic = max(spot - strike, 0.0)
        if opt == "p":
            intrinsic = max(strike - spot, 0.0)
        return intrinsic
    if spot <= 0.0 or strike <= 0.0 or sigma <= 0.0:
        return float("nan")

    if abs(1.0 - beta) < 1.0e-6:
        from black_scholes import bs_price

        return bs_price(opt, spot, strike, rate, time, sigma, q)

    m = 2.0 * (1.0 - beta)
    b = rate - q
    if abs(b) < 1.0e-12:
        kappa = 2.0 / (sigma * sigma * m * m * time)
    else:
        denom = sigma * sigma * m * (math.exp(b * m * time) - 1.0)
        if denom == 0.0:
            kappa = 2.0 / (sigma * sigma * m * m * time)
        else:
            kappa = 2.0 * b / denom

    x = kappa * (spot ** m) * math.exp(b * m * time)
    y = kappa * (strike ** m)
    df1 = 2.0 + 2.0 / m
    df2 = 2.0 / m

    cdf1 = ncx2.cdf(2.0 * y, df1, 2.0 * x)
    cdf2 = ncx2.cdf(2.0 * x, df2, 2.0 * y)
    call = spot * math.exp(-q * time) * (1.0 - cdf1) - strike * math.exp(-rate * time) * cdf2
    if opt == "c":
        return max(call, 0.0)
    put = call - spot * math.exp(-q * time) + strike * math.exp(-rate * time)
    return max(put, 0.0)


def fit_cev(
    spot: float,
    rate: float,
    time: float,
    strikes: Iterable[float],
    prices: Iterable[float],
    option_types: Iterable[str],
    beta: float = 0.7,
    fit_beta: bool = True,
    q: float = 0.0,
    sigma0: float | None = None,
) -> dict:
    """Fit CEV parameters to option prices."""
    strikes = np.asarray(list(strikes), dtype=float)
    prices = np.asarray(list(prices), dtype=float)
    option_types = np.asarray(list(option_types), dtype=str)

    if sigma0 is None:
        sigma0 = 0.2

    if fit_beta:
        x0 = np.array([sigma0, beta], dtype=float)
        lower = np.array([1.0e-6, 0.01], dtype=float)
        upper = np.array([5.0, 1.0], dtype=float)
    else:
        x0 = np.array([sigma0], dtype=float)
        lower = np.array([1.0e-6], dtype=float)
        upper = np.array([5.0], dtype=float)

    def resid(params: np.ndarray) -> np.ndarray:
        if fit_beta:
            sigma, beta_val = params
        else:
            sigma = params[0]
            beta_val = beta
        out = []
        for k, price, opt in zip(strikes, prices, option_types):
            model = cev_price(opt, spot, k, rate, time, sigma, beta_val, q)
            if not np.isfinite(model):
                model = 0.0
            out.append(model - price)
        return np.asarray(out, dtype=float)

    res = least_squares(resid, x0, bounds=(lower, upper), max_nfev=5000)
    if fit_beta:
        sigma, beta_fit = res.x
    else:
        sigma = res.x[0]
        beta_fit = beta
    rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")

    return {
        "sigma": float(sigma),
        "beta": float(beta_fit),
        "rmse": rmse,
    }
