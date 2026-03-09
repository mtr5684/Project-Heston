"""Option Greeks: analytical Black-Scholes and numerical bump-and-reprice."""

import numpy as np
from models.black_scholes import bs_d1, bs_d2, bs_call_price, bs_put_price
from utils.math_utils import norm_cdf, norm_pdf


# ---------------------------------------------------------------------------
# Analytical Black-Scholes Greeks
# ---------------------------------------------------------------------------

def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Analytical BS delta.

    Call: N(d1)
    Put:  N(d1) - 1
    """
    d1 = bs_d1(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm_cdf(d1))
    return float(norm_cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical BS gamma (same for call and put).

    Gamma = n(d1) / (S * sigma * sqrt(T))
    """
    d1 = bs_d1(S, K, T, r, sigma)
    return float(norm_pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical BS vega (same for call and put).

    Vega = S * n(d1) * sqrt(T)
    Returns vega for a 1-point (100%) move in sigma. Divide by 100 for 1% move.
    """
    d1 = bs_d1(S, K, T, r, sigma)
    return float(S * norm_pdf(d1) * np.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Analytical BS theta (per year).

    Call: -(S*n(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)
    Put:  -(S*n(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2)
    """
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    common = -S * norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return float(common - r * K * np.exp(-r * T) * norm_cdf(d2))
    return float(common + r * K * np.exp(-r * T) * norm_cdf(-d2))


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Analytical BS rho.

    Call: K * T * exp(-rT) * N(d2)
    Put: -K * T * exp(-rT) * N(-d2)
    """
    d2 = bs_d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm_cdf(d2))
    return float(-K * T * np.exp(-r * T) * norm_cdf(-d2))


# ---------------------------------------------------------------------------
# Model-agnostic numerical Greeks (bump-and-reprice)
# ---------------------------------------------------------------------------

def numerical_delta(
    pricing_fn: callable,
    S: float,
    bump: float = 0.01,
    **pricing_kwargs,
) -> float:
    """Numerical delta via central finite difference.

    delta = (V(S*(1+bump)) - V(S*(1-bump))) / (2 * bump * S)

    pricing_fn signature: pricing_fn(S0=..., **kwargs) -> float
    """
    dS = bump * S
    price_up = pricing_fn(S0=S + dS, **pricing_kwargs)
    price_down = pricing_fn(S0=S - dS, **pricing_kwargs)
    return (price_up - price_down) / (2.0 * dS)


def numerical_gamma(
    pricing_fn: callable,
    S: float,
    bump: float = 0.01,
    **pricing_kwargs,
) -> float:
    """Numerical gamma via central finite difference.

    gamma = (V(S+dS) - 2*V(S) + V(S-dS)) / (dS^2)
    """
    dS = bump * S
    price_up = pricing_fn(S0=S + dS, **pricing_kwargs)
    price_mid = pricing_fn(S0=S, **pricing_kwargs)
    price_down = pricing_fn(S0=S - dS, **pricing_kwargs)
    return (price_up - 2.0 * price_mid + price_down) / (dS**2)


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    print("=== Analytical BS Greeks (Call) ===")
    print(f"Delta: {bs_delta(S, K, T, r, sigma, 'call'):.6f}")
    print(f"Gamma: {bs_gamma(S, K, T, r, sigma):.6f}")
    print(f"Vega:  {bs_vega(S, K, T, r, sigma):.4f}")
    print(f"Theta: {bs_theta(S, K, T, r, sigma, 'call'):.4f}")
    print(f"Rho:   {bs_rho(S, K, T, r, sigma, 'call'):.4f}")

    print("\n=== Analytical BS Greeks (Put) ===")
    print(f"Delta: {bs_delta(S, K, T, r, sigma, 'put'):.6f}")
    print(f"Theta: {bs_theta(S, K, T, r, sigma, 'put'):.4f}")
    print(f"Rho:   {bs_rho(S, K, T, r, sigma, 'put'):.4f}")

    # Verify numerical delta matches analytical for BS
    def _bs_call_wrapper(S0, K=K, T=T, r=r, sigma=sigma):
        return bs_call_price(S0, K, T, r, sigma)

    num_d = numerical_delta(_bs_call_wrapper, S)
    ana_d = bs_delta(S, K, T, r, sigma, "call")
    print(f"\n=== Numerical vs Analytical Delta ===")
    print(f"Numerical: {num_d:.6f}")
    print(f"Analytical: {ana_d:.6f}")
    print(f"Difference: {abs(num_d - ana_d):.2e}")
