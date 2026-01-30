#!/usr/bin/env python3
"""
SABR utilities (Hagan 2002 lognormal approximation).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import least_squares


def sabr_iv_hagan(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Return Hagan lognormal SABR implied vol."""
    if f <= 0.0 or k <= 0.0 or t <= 0.0:
        return float("nan")
    if alpha <= 0.0 or nu <= 0.0 or not (-0.999 < rho < 0.999):
        return float("nan")
    if not (0.0 <= beta <= 1.0):
        return float("nan")

    log_fk = np.log(f / k)
    fk = f * k
    one_minus_b = 1.0 - beta
    fk_pow = fk ** (one_minus_b * 0.5)
    if fk_pow <= 0:
        return float("nan")

    if abs(log_fk) < 1e-12:
        # ATM formula
        f_pow = f ** one_minus_b
        if f_pow <= 0:
            return float("nan")
        term1 = ((one_minus_b ** 2) / 24.0) * (alpha ** 2) / (f_pow ** 2)
        term2 = (rho * beta * nu * alpha) / (4.0 * f_pow)
        term3 = ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
        return (alpha / f_pow) * (1.0 + (term1 + term2 + term3) * t)

    z = (nu / alpha) * fk_pow * log_fk
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    if xz == 0.0:
        return float("nan")
    denom = fk_pow * (1.0 + (one_minus_b ** 2 / 24.0) * log_fk ** 2 +
                      (one_minus_b ** 4 / 1920.0) * log_fk ** 4)
    if denom <= 0:
        return float("nan")
    term1 = alpha / denom
    term2 = z / xz
    f_pow = (fk ** one_minus_b)
    term3 = ((one_minus_b ** 2) / 24.0) * (alpha ** 2) / f_pow
    term4 = (rho * beta * nu * alpha) / (4.0 * fk_pow)
    term5 = ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
    return term1 * term2 * (1.0 + (term3 + term4 + term5) * t)


def fit_sabr(
    f: float,
    k: np.ndarray,
    t: float,
    iv: np.ndarray,
    beta: float = 1.0,
    fit_beta: bool = False,
) -> Dict[str, float]:
    """Fit SABR parameters to implied vols."""
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if k.size == 0 or iv.size == 0:
        return {"alpha": float("nan"), "beta": beta, "rho": float("nan"),
                "nu": float("nan"), "rmse": float("nan")}

    idx = np.argmin(np.abs(k - f))
    atm_iv = float(iv[idx]) if np.isfinite(iv[idx]) else float(np.nanmean(iv))
    if not np.isfinite(atm_iv) or atm_iv <= 0:
        atm_iv = 0.2
    alpha0 = atm_iv * (f ** (1.0 - beta))

    alpha_upper = max(10.0, alpha0 * 10.0)
    if fit_beta:
        x0 = np.array([alpha0, beta, 0.0, 0.5], dtype=float)
        lower = np.array([1e-8, 0.0, -0.999, 1e-6], dtype=float)
        upper = np.array([alpha_upper, 1.0, 0.999, 10.0], dtype=float)

        def resid(params: np.ndarray) -> np.ndarray:
            a, b, r, n = params
            iv_fit = np.array([sabr_iv_hagan(f, kk, t, a, b, r, n) for kk in k])
            return iv_fit - iv
    else:
        x0 = np.array([alpha0, 0.0, 0.5], dtype=float)
        lower = np.array([1e-8, -0.999, 1e-6], dtype=float)
        upper = np.array([alpha_upper, 0.999, 10.0], dtype=float)

        def resid(params: np.ndarray) -> np.ndarray:
            a, r, n = params
            iv_fit = np.array([sabr_iv_hagan(f, kk, t, a, beta, r, n) for kk in k])
            return iv_fit - iv

    res = least_squares(resid, x0, bounds=(lower, upper), max_nfev=5000)
    if fit_beta:
        alpha, beta, rho, nu = res.x
    else:
        alpha, rho, nu = res.x
    rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu, "rmse": rmse}
