import numpy as np
from scipy import stats
from scipy.optimize import brentq
from option_pricing import crr_binomial_option, bjerksund_stensland_1993


def bs_price(option_type: str, spot: float, strike: float,
             rate: float, time: float, vol: float, q: float) -> float:
    """Return Black-Scholes price for call/put/straddle."""
    if vol <= 0:
        return np.nan
    sqrt_time = np.sqrt(time)
    d1 = (np.log(spot / strike) + (rate - q + 0.5 * vol**2) * time) / (vol * sqrt_time)
    d2 = d1 - vol * sqrt_time
    disc_q = np.exp(-q * time)
    disc_r = np.exp(-rate * time)
    call_price = spot * disc_q * stats.norm.cdf(d1) - strike * disc_r * stats.norm.cdf(d2)
    put_price = strike * disc_r * stats.norm.cdf(-d2) - spot * disc_q * stats.norm.cdf(-d1)
    opt = option_type.lower()
    if opt == "c":
        return call_price
    if opt == "p":
        return put_price
    if opt == "straddle":
        return call_price + put_price
    raise ValueError("option_type must be 'c', 'p', or 'straddle'")

def implied_vol(option_type: str, price: float, spot: float, strike: float,
                rate: float, time: float, q: float,
                model: str = "european", tree_steps: int = 200,
                american_method: str = "bjerksund") -> float:
    """Return implied volatility from a call/put/straddle price."""
    model_lc = model.lower()

    def model_price(vol: float) -> float:
        if model_lc == "american":
            method = american_method.lower()
            if method == "bjerksund":
                if option_type == "straddle":
                    call = bjerksund_stensland_1993(spot, strike, rate, q, vol, time, "call",
                                                   fallback_steps=tree_steps)
                    put = bjerksund_stensland_1993(spot, strike, rate, q, vol, time, "put",
                                                  fallback_steps=tree_steps)
                    return call + put
                if option_type == "c":
                    return bjerksund_stensland_1993(spot, strike, rate, q, vol, time, "call",
                                                   fallback_steps=tree_steps)
                if option_type == "p":
                    return bjerksund_stensland_1993(spot, strike, rate, q, vol, time, "put",
                                                   fallback_steps=tree_steps)
                raise ValueError("option_type must be 'c', 'p', or 'straddle'")
            if option_type == "straddle":
                call = crr_binomial_option(spot, strike, rate, q, vol, time, tree_steps,
                                           option_type="call", exercise="american")
                put = crr_binomial_option(spot, strike, rate, q, vol, time, tree_steps,
                                          option_type="put", exercise="american")
                return call + put
            if option_type == "c":
                return crr_binomial_option(spot, strike, rate, q, vol, time, tree_steps,
                                           option_type="call", exercise="american")
            if option_type == "p":
                return crr_binomial_option(spot, strike, rate, q, vol, time, tree_steps,
                                           option_type="put", exercise="american")
            raise ValueError("option_type must be 'c', 'p', or 'straddle'")
        return bs_price(option_type, spot, strike, rate, time, vol, q)

    def objective(vol: float) -> float:
        try:
            return model_price(vol) - price
        except (ValueError, OverflowError):
            return np.nan

    # intrinsic check (especially important for American)
    if option_type == "c":
        intrinsic = max(spot - strike, 0.0)
    elif option_type == "p":
        intrinsic = max(strike - spot, 0.0)
    else:
        intrinsic = max(spot - strike, 0.0) + max(strike - spot, 0.0)
    if price <= intrinsic + 1e-12:
        return 0.0

    # bracket search
    if model_lc == "american":
        min_vol = abs(rate - q) * np.sqrt(time / max(tree_steps, 1))
        lo = max(1e-4, min_vol * 1.001)
        hi = 10.0
    else:
        lo = 1e-6
        hi = 5.0
    try:
        f_lo = objective(lo)
        while not np.isfinite(f_lo) and lo < hi:
            lo *= 2.0
            f_lo = objective(lo)
        f_hi = objective(hi)
        attempts = 0
        while (not np.isfinite(f_hi) or f_lo * f_hi > 0) and attempts < 5:
            hi *= 2.0
            f_hi = objective(hi)
            attempts += 1
        if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
            return np.nan
        return brentq(objective, lo, hi)
    except ValueError:
        return np.nan
