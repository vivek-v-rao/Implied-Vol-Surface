# Models

This page summarizes the option pricing and smile/surface models implemented in this repo.

## Black-Scholes (European)

For spot *S*, strike *K*, rate *r*, dividend yield *q*, volatility *σ*, and time *T*:

- Forward: *F = S·e^{(r−q)T}*
- *d1 = [ln(S/K) + (r−q + 0.5σ²)T] / (σ√T)*
- *d2 = d1 − σ√T*
- Call: *C = S·e^{−qT}·N(d1) − K·e^{−rT}·N(d2)*
- Put: *P = C − S·e^{−qT} + K·e^{−rT}*

## SABR (Hagan lognormal approximation)

Parameters: *α* (vol level), *β* (elasticity), *ρ* (correlation), *ν* (vol of vol).

Given forward *F*, strike *K*, and *T*:

- *z = (ν/α)·(F·K)^{(1−β)/2}·ln(F/K)*
- *x(z) = ln[(√(1−2ρz+z²) + z − ρ)/(1−ρ)]*
- Implied vol (lognormal): *σ_{BS}(F,K) = [α / ((F·K)^{(1−β)/2}·(1 + ((1−β)²/24)ln²(F/K) + ((1−β)^4/1920)ln^4(F/K)))] · (z/x(z)) · [1 + O(T)]*

ATM expansion (used in code) follows Hagan et al. (2002).

## SVI (per-expiry smile)

Total variance *w(k)* as a function of log-moneyness *k = ln(K/F)*:

- *w(k) = a + b·[ρ·(k−m) + √((k−m)² + σ²)]*

Parameters: *a, b, ρ, m, σ*.

## SSVI (surface SVI)

Define ATM total variance *θ(T)* and log-moneyness *k = ln(K/F)*:

- *w(k,θ) = 0.5·θ·[1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + 1 − ρ²)]*
- *φ(θ) = η / θ^γ*

Parameters: *ρ, η, γ* and *θ*.

### eSSVI (parametric θ(T))

- *θ(T) = a + b·T^c*

Parameters: *a, b, c, ρ, η, γ*.

## CEV (European)

Under the CEV process *dS = (r−q)S dt + σ S^β dW* with *β < 1*, the European option price can be expressed via the noncentral χ² distribution.

The implementation uses the standard noncentral χ² closed-form (Cox, 1975).

