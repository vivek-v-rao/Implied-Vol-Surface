#!/usr/bin/env python3
"""
SSVI (surface SVI) utilities.

Implements the SSVI total variance:
  w(k, theta) = 0.5 * theta * (1 + rho * phi(theta) * k +
                               sqrt((phi(theta) * k + rho)**2 + 1 - rho**2))
where k is log-moneyness and theta is total variance at the money.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import least_squares


def ssvi_total_variance(
    k: np.ndarray,
    theta: float | np.ndarray,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Compute SSVI total variance for log-moneyness k."""
    theta_arr = np.asarray(theta, dtype=float)
    if np.any(theta_arr <= 0.0):
        return np.full_like(k, np.nan, dtype=float)
    if not (-0.999 <= rho <= 0.999):
        return np.full_like(k, np.nan, dtype=float)
    if eta <= 0.0 or gamma <= 0.0:
        return np.full_like(k, np.nan, dtype=float)
    phi = eta / (theta_arr ** gamma)
    term = phi * k + rho
    return 0.5 * theta_arr * (1.0 + rho * phi * k + np.sqrt(term * term + 1.0 - rho * rho))


def fit_ssvi_theta(
    k: np.ndarray,
    w: np.ndarray,
    rho: float,
    eta: float,
    gamma: float,
) -> Dict[str, float]:
    """Fit theta for fixed rho/eta/gamma via least squares."""
    thetas = np.linspace(1e-4, 2.0, 400)
    best = None
    for theta in thetas:
        w_fit = ssvi_total_variance(k, theta, rho, eta, gamma)
        if np.any(~np.isfinite(w_fit)):
            continue
        rmse = np.sqrt(np.mean((w_fit - w) ** 2))
        if best is None or rmse < best["rmse"]:
            best = {"theta": theta, "rmse": rmse}
    if best is None:
        return {"theta": float("nan"), "rmse": float("nan")}
    return best


def fit_ssvi(
    k: np.ndarray,
    w: np.ndarray,
    theta0: float | None = None,
    rho0: float = -0.2,
    eta0: float = 0.5,
    gamma0: float = 0.5,
) -> Dict[str, float]:
    """Fit theta, rho, eta, gamma by least squares."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if theta0 is None:
        theta0 = float(np.nanmedian(w)) if np.isfinite(np.nanmedian(w)) else 0.1
        if theta0 <= 0:
            theta0 = 0.1

    def loss(params: np.ndarray) -> np.ndarray:
        theta, rho, eta, gamma = params
        w_fit = ssvi_total_variance(k, theta, rho, eta, gamma)
        return w_fit - w

    x0 = np.array([theta0, rho0, eta0, gamma0], dtype=float)
    lower = np.array([1e-6, -0.999, 1e-6, 0.01], dtype=float)
    upper = np.array([10.0, 0.999, 10.0, 5.0], dtype=float)
    res = least_squares(loss, x0, bounds=(lower, upper), max_nfev=5000)
    theta, rho, eta, gamma = res.x
    rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
    return {"theta": theta, "rho": rho, "eta": eta, "gamma": gamma, "rmse": rmse}


def essvi_total_variance(
    k: np.ndarray,
    t: np.ndarray,
    a: float,
    b: float,
    c: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Compute eSSVI total variance for log-moneyness k and tenor t."""
    t = np.asarray(t, dtype=float)
    theta = a + b * np.power(t, c)
    if np.any(theta <= 0.0):
        return np.full_like(k, np.nan, dtype=float)
    return ssvi_total_variance(k, theta, rho, eta, gamma)


def fit_essvi(
    k: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    a0: float | None = None,
    b0: float = 0.1,
    c0: float = 0.5,
    rho0: float = -0.2,
    eta0: float = 0.5,
    gamma0: float = 0.5,
) -> Dict[str, float]:
    """Fit eSSVI parameters with constant rho/eta/gamma."""
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    if a0 is None:
        a0 = float(np.nanmedian(w)) if np.isfinite(np.nanmedian(w)) else 0.1
        if a0 <= 0:
            a0 = 0.1

    def loss(params: np.ndarray) -> np.ndarray:
        a, b, c, rho, eta, gamma = params
        w_fit = essvi_total_variance(k, t, a, b, c, rho, eta, gamma)
        return w_fit - w

    x0 = np.array([a0, b0, c0, rho0, eta0, gamma0], dtype=float)
    lower = np.array([1e-6, 0.0, 0.05, -0.999, 1e-6, 0.01], dtype=float)
    upper = np.array([10.0, 10.0, 5.0, 0.999, 10.0, 5.0], dtype=float)
    res = least_squares(loss, x0, bounds=(lower, upper), max_nfev=8000)
    a, b, c, rho, eta, gamma = res.x
    rmse = float(np.sqrt(np.mean(res.fun ** 2))) if res.fun.size else float("nan")
    return {"a": a, "b": b, "c": c, "rho": rho, "eta": eta, "gamma": gamma, "rmse": rmse}
