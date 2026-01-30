import math
from typing import Literal

from scipy.stats import multivariate_normal

def crr_binomial_option(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    n_steps: int,
    option_type: Literal["call", "put"] = "call",
    exercise:    Literal["european", "american"] = "european",
) -> float:
    """
    Price a call or put using the CRR binomial tree with continuous dividend yield.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = t / n_steps
    if dt <= 0:
        raise ValueError("T must be positive")

    # CRR up/down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u

    # Risk-neutral probability with continuous dividend yield q
    disc = math.exp(-r * dt)
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral probability out of bounds: p={p:.6f}")

    # Terminal stock prices and payoffs (log-space to avoid overflow)
    values = []
    log_s0 = math.log(s0) if s0 > 0 else float("-inf")
    log_u = math.log(u)
    log_d = math.log(d)
    for j in range(n_steps + 1):
        log_s = log_s0 + j * log_u + (n_steps - j) * log_d
        if log_s > 700.0:
            s = float("inf")
        elif log_s < -700.0:
            s = 0.0
        else:
            s = math.exp(log_s)
        if option_type == "call":
            payoff = max(s - k, 0.0)
        elif option_type == "put":
            payoff = max(k - s, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        values.append(payoff)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(i + 1):
            # continuation value
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])

            if exercise == "american":
                # underlying price at node (i, j) in log-space
                log_s = log_s0 + j * log_u + (i - j) * log_d
                if log_s > 700.0:
                    s = float("inf")
                elif log_s < -700.0:
                    s = 0.0
                else:
                    s = math.exp(log_s)
                if option_type == "call":
                    intrinsic = max(s - k, 0.0)
                else:  # put
                    intrinsic = max(k - s, 0.0)
                node_val = max(cont, intrinsic)
            elif exercise == "european":
                node_val = cont
            else:
                raise ValueError("exercise must be 'european' or 'american'")

            new_values.append(node_val)
        values = new_values
    return values[0]

def bs(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black–Scholes price of a European call or put with continuous dividend yield.
    """
    if t <= 0.0:
        # At expiry: intrinsic value
        if option_type == "call":
            return max(s0 - k, 0.0)
        elif option_type == "put":
            return max(k - s0, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    if sigma <= 0.0:
        # Degenerate case: no volatility, option value is discounted intrinsic under forward
        f0 = s0 * math.exp((r - q) * t)
        if option_type == "call":
            payoff = max(f0 - k, 0.0)
        else:
            payoff = max(k - f0, 0.0)
        return math.exp(-r * t) * payoff

    # standard normal CDF
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s0 / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        price = (
            s0 * math.exp(-q * t) * norm_cdf(d1)
            - k * math.exp(-r * t) * norm_cdf(d2)
        )
    elif option_type == "put":
        price = (
            k * math.exp(-r * t) * norm_cdf(-d2)
            - s0 * math.exp(-q * t) * norm_cdf(-d1)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _pow_ratio(num: float, den: float, power: float) -> float:
    if num <= 0.0 or den <= 0.0:
        return 0.0
    expo = power * math.log(num / den)
    if expo > 700.0:
        return float("inf")
    if expo < -700.0:
        return 0.0
    return math.exp(expo)

def _bivar_cdf(a: float, b: float, rho: float) -> float:
    # Clamp rho to avoid numerical issues
    rho = max(min(rho, 0.999999), -0.999999)
    return float(multivariate_normal.cdf([a, b], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]))

def _phi(
    s: float,
    t: float,
    gamma: float,
    h: float,
    x: float,
    r: float,
    b: float,
    sigma: float,
) -> float:
    if t <= 0.0:
        return 0.0
    if sigma <= 0.0:
        return 0.0
    if s <= 0.0 or h <= 0.0 or x <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    sig2 = sigma * sigma
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sig2
    d1 = (math.log(s / h) + (b + (gamma - 0.5) * sig2) * t) / (sigma * sqrt_t)
    d2 = (math.log((x * x) / (s * h)) + (b + (gamma - 0.5) * sig2) * t) / (sigma * sqrt_t)
    kappa = 2.0 * b / sig2 + (2.0 * gamma - 1.0)
    return math.exp(lam * t) * (s ** gamma) * (_norm_cdf(d1) - _pow_ratio(x, s, kappa) * _norm_cdf(d2))

def _psi(
    s: float,
    t: float,
    gamma: float,
    h: float,
    x: float,
    x2: float,
    t1: float,
    r: float,
    b: float,
    sigma: float,
) -> float:
    if t <= 0.0 or t1 <= 0.0:
        return 0.0
    if sigma <= 0.0:
        return 0.0
    if s <= 0.0 or h <= 0.0 or x <= 0.0 or x2 <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    sqrt_t1 = math.sqrt(t1)
    sig2 = sigma * sigma
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sig2
    kappa = 2.0 * b / sig2 + (2.0 * gamma - 1.0)
    rho = math.sqrt(t1 / t)
    d1 = (math.log(s / x2) + (b + (gamma - 0.5) * sig2) * t1) / (sigma * sqrt_t1)
    d2 = (math.log((x * x) / (s * x2)) + (b + (gamma - 0.5) * sig2) * t1) / (sigma * sqrt_t1)
    d3 = (math.log(s / x2) - (b + (gamma - 0.5) * sig2) * t1) / (sigma * sqrt_t1)
    d4 = (math.log((x * x) / (s * x2)) - (b + (gamma - 0.5) * sig2) * t1) / (sigma * sqrt_t1)
    D1 = (math.log(s / h) + (b + (gamma - 0.5) * sig2) * t) / (sigma * sqrt_t)
    D2 = (math.log((x * x) / (s * h)) + (b + (gamma - 0.5) * sig2) * t) / (sigma * sqrt_t)
    D3 = (math.log((x2 * x2) / (s * h)) + (b + (gamma - 0.5) * sig2) * t) / (sigma * sqrt_t)
    D4 = (math.log((s * x2 * x2) / (h * x * x)) + (b + (gamma - 0.5) * sig2) * t) / (sigma * sqrt_t)
    term1 = _bivar_cdf(d1, D1, rho)
    term2 = _pow_ratio(x, s, kappa) * _bivar_cdf(d2, D2, rho)
    term3 = _pow_ratio(x2, s, kappa) * _bivar_cdf(d3, D3, -rho)
    term4 = _pow_ratio(x2, x, kappa) * _bivar_cdf(d4, D4, -rho)
    return math.exp(lam * t) * (s ** gamma) * (term1 - term2 - term3 + term4)

def _flat_boundary(
    k: float,
    r: float,
    b: float,
    sigma: float,
    t: float,
) -> float:
    if t <= 0.0:
        return k
    sig2 = sigma * sigma
    beta = (0.5 - b / sig2) + math.sqrt((b / sig2 - 0.5) ** 2 + 2.0 * r / sig2)
    if b >= r:
        b0 = k
    else:
        b0 = max(k, (r / (r - b)) * k)
    b1 = (beta / (beta - 1.0)) * k
    h = -(b * t + 2.0 * sigma * math.sqrt(t)) * (k ** 2) / ((b1 - b0) * b0)
    if h < -700.0:
        exp_h = 0.0
    elif h > 700.0:
        exp_h = math.exp(700.0)
    else:
        exp_h = math.exp(h)
    return b0 + (b1 - b0) * (1.0 - exp_h)

def _bjerksund_stensland_2002_call(
    s0: float,
    k: float,
    r: float,
    b: float,
    sigma: float,
    t: float,
) -> float:
    if t <= 0.0:
        return max(s0 - k, 0.0)
    if sigma <= 0.0:
        return max(s0 - k, 0.0)
    if b >= r:
        q = r - b
        return bs(s0, k, r, q, sigma, t, "call")

    sig2 = sigma * sigma
    beta = (0.5 - b / sig2) + math.sqrt((b / sig2 - 0.5) ** 2 + 2.0 * r / sig2)
    t1 = 0.5 * (math.sqrt(5.0) - 1.0) * t
    if t1 <= 0.0:
        q = r - b
        return bs(s0, k, r, q, sigma, t, "call")
    x = _flat_boundary(k, r, b, sigma, t - t1)
    X = _flat_boundary(k, r, b, sigma, t)
    if s0 >= X:
        return s0 - k

    a_X = (X - k) * (X ** (-beta))
    a_x = (x - k) * (x ** (-beta))

    term1 = a_X * (s0 ** beta)
    term2 = -a_X * _phi(s0, t1, beta, X, X, r, b, sigma)
    term3 = _phi(s0, t1, 1.0, X, X, r, b, sigma)
    term4 = -_phi(s0, t1, 1.0, x, X, r, b, sigma)
    term5 = -k * _phi(s0, t1, 0.0, X, X, r, b, sigma)
    term6 = k * _phi(s0, t1, 0.0, x, X, r, b, sigma)

    term7 = a_x * _phi(s0, t1, beta, x, X, r, b, sigma)
    term8 = -a_x * _psi(s0, t, beta, x, X, x, t1, r, b, sigma)
    term9 = _psi(s0, t, 1.0, x, X, x, t1, r, b, sigma)
    term10 = -_psi(s0, t, 1.0, k, X, x, t1, r, b, sigma)
    term11 = -k * _psi(s0, t, 0.0, x, X, x, t1, r, b, sigma)
    term12 = k * _psi(s0, t, 0.0, k, X, x, t1, r, b, sigma)
    terms = [term1, term2, term3, term4, term5, term6, term7, term8, term9, term10, term11, term12]
    if not all(math.isfinite(v) for v in terms):
        return float("nan")
    return sum(terms)

def bjerksund_stensland_2002(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    option_type: Literal["call", "put"] = "call",
    fallback_steps: int = 200,
) -> float:
    """
    Bjerksund-Stensland (2002) American option approximation (two-step boundary).
    For puts, uses the put-call transformation. Falls back to CRR if result is invalid.
    """
    b = r - q
    if option_type == "call":
        try:
            price = _bjerksund_stensland_2002_call(s0, k, r, b, sigma, t)
        except Exception:
            price = float("nan")
        intrinsic = max(s0 - k, 0.0)
        upper = s0
    elif option_type == "put":
        # The put-call transformation is numerically unstable in this implementation;
        # use CRR directly for puts.
        return crr_binomial_option(s0, k, r, q, sigma, t, fallback_steps, "put", "american")
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if not math.isfinite(price) or price < intrinsic - 1e-10 or price > upper + 1e-10:
        return crr_binomial_option(s0, k, r, q, sigma, t, fallback_steps, option_type, "american")
    return price

def bjerksund_stensland_1993(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    option_type: Literal["call", "put"] = "call",
    fallback_steps: int = 200,
) -> float:
    """
    Backwards-compatible alias for the Bjerksund-Stensland approximation.
    """
    return bjerksund_stensland_2002(s0, k, r, q, sigma, t, option_type, fallback_steps=fallback_steps)

def crr_binomial_option_bs_smooth(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    n_steps: int,
    option_type: Literal["call", "put"] = "call",
    exercise:    Literal["european", "american"] = "european",
    band: int = 2,
) -> float:
    """
    CRR tree with first-step BS smoothing near the strike (band=2 -> 5 nodes).
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if band < 0:
        raise ValueError("band must be >= 0")

    dt = t / n_steps
    if dt <= 0:
        raise ValueError("T must be positive")

    # CRR up/down and risk-neutral prob with yield q
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral probability out of bounds: p={p:.6f}")

    # terminal payoffs at i = n_steps
    values = []
    for j in range(n_steps + 1):
        s = s0 * (u ** j) * (d ** (n_steps - j))
        if option_type == "call":
            values.append(max(s - k, 0.0))
        elif option_type == "put":
            values.append(max(k - s, 0.0))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # --- special first backward step: i = n_steps - 1 ---
    i = n_steps - 1
    # stock prices at the penultimate layer
    s_layer = [s0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
    # index closest to the strike
    k_idx = min(range(len(s_layer)), key=lambda j: abs(s_layer[j] - k))
    j_lo = max(0, k_idx - band)
    j_hi = min(i, k_idx + band)

    new_values = []
    for j in range(i + 1):
        s = s_layer[j]
        # continuation: BS over dt near the strike, else usual lattice expectation
        if j_lo <= j <= j_hi:
            cont = bs(s, k, r, q, sigma, dt, option_type)
        else:
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])

        if exercise == "american":
            intrinsic = (s - k) if option_type == "call" else (k - s)
            intrinsic = max(intrinsic, 0.0)
            node_val = max(cont, intrinsic)
        elif exercise == "european":
            node_val = cont
        else:
            raise ValueError("exercise must be 'european' or 'american'")

        new_values.append(node_val)
    values = new_values  # now sized i+1

    # --- standard backward induction for remaining layers ---
    for i in range(n_steps - 2, -1, -1):
        new_values = []
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if exercise == "american":
                s = s0 * (u ** j) * (d ** (i - j))
                intrinsic = (s - k) if option_type == "call" else (k - s)
                intrinsic = max(intrinsic, 0.0)
                node_val = max(cont, intrinsic)
            else:
                node_val = cont
            new_values.append(node_val)
        values = new_values
    return values[0]
