#!/usr/bin/env python3
"""
SVI utilities.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Compute SVI total variance for log-moneyness k."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(k: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Fit SVI parameters to total variance."""
    a0 = max(np.mean(w), 1e-6)
    b0 = 0.1
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.1

    def resid(params: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        w_fit = svi_total_variance(k, a, b, rho, m, sigma)
        return w_fit - w

    bounds = (
        np.array([0.0, 0.0, -0.999, -10.0, 1e-6]),
        np.array([10.0, 10.0, 0.999, 10.0, 5.0]),
    )
    x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float)
    res = least_squares(resid, x0, bounds=bounds, max_nfev=5000)
    a, b, rho, m, sigma = res.x
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "rmse": np.sqrt(np.mean(res.fun ** 2))}
