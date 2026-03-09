"""Heston stochastic volatility model — Monte Carlo simulation."""

from dataclasses import dataclass
import numpy as np
from utils.math_utils import generate_correlated_bm


@dataclass
class HestonParams:
    """Heston model parameters.

    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_t^1
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^2
    Corr(dW^1, dW^2) = rho
    """
    kappa: float    # mean-reversion speed of variance
    theta: float    # long-run variance level
    xi: float       # volatility of variance (vol-of-vol)
    rho: float      # correlation between stock and variance Brownian motions
    v0: float       # initial variance

    def feller_condition(self) -> bool:
        """Check if 2*kappa*theta > xi^2 (variance stays positive)."""
        return 2.0 * self.kappa * self.theta > self.xi ** 2

    def as_array(self) -> np.ndarray:
        """Return parameters as array [kappa, theta, xi, rho, v0]."""
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        """Construct from array [kappa, theta, xi, rho, v0]."""
        return cls(kappa=x[0], theta=x[1], xi=x[2], rho=x[3], v0=x[4])


def simulate_heston(
    S0: float,
    r: float,
    params: HestonParams,
    T: float,
    n_steps: int = 252,
    n_paths: int = 100_000,
    scheme: str = "reflection",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths using Euler-Maruyama.

    Schemes for handling negative variance:
        - "reflection": v = |v| after each step
        - "absorption": v = max(v, 0) after each step
        - "full_truncation": use v^+ in drift and diffusion, store raw v

    Returns:
        S: shape (n_paths, n_steps + 1) — stock price paths
        v: shape (n_paths, n_steps + 1) — variance paths
    """
    dt = T / n_steps
    kappa, theta, xi, rho, v0 = (
        params.kappa, params.theta, params.xi, params.rho, params.v0
    )

    dW1, dW2 = generate_correlated_bm(n_paths, n_steps, dt, rho, seed=seed)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(n_steps):
        v_curr = v[:, t]

        if scheme == "full_truncation":
            v_pos = np.maximum(v_curr, 0.0)
        elif scheme == "reflection":
            v_pos = np.abs(v_curr)
        else:  # absorption
            v_pos = np.maximum(v_curr, 0.0)

        sqrt_v = np.sqrt(v_pos)

        # Log-Euler for S (avoids negative stock prices)
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_pos) * dt + sqrt_v * dW1[:, t]
        )

        # Euler for v
        v_next = v_curr + kappa * (theta - v_curr) * dt + xi * sqrt_v * dW2[:, t]

        if scheme == "reflection":
            v[:, t + 1] = np.abs(v_next)
        elif scheme == "absorption":
            v[:, t + 1] = np.maximum(v_next, 0.0)
        else:  # full_truncation: store raw value
            v[:, t + 1] = v_next

    return S, v


def heston_mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    n_steps: int = 252,
    n_paths: int = 100_000,
    option_type: str = "call",
    scheme: str = "reflection",
    seed: int = 42,
) -> float:
    """Price a European option under Heston via Monte Carlo."""
    S, _ = simulate_heston(S0, r, params, T, n_steps, n_paths, scheme, seed)
    S_T = S[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    return float(np.exp(-r * T) * np.mean(payoffs))


def heston_mc_price_with_se(
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    n_steps: int = 252,
    n_paths: int = 100_000,
    option_type: str = "call",
    scheme: str = "reflection",
    seed: int = 42,
) -> tuple[float, float]:
    """Returns (price, standard_error)."""
    S, _ = simulate_heston(S0, r, params, T, n_steps, n_paths, scheme, seed)
    S_T = S[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    se = float(np.std(discounted) / np.sqrt(n_paths))
    return price, se


if __name__ == "__main__":
    # Test with typical parameters
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print(f"Feller condition: {params.feller_condition()}")
    print(f"  2*kappa*theta = {2 * params.kappa * params.theta:.4f}")
    print(f"  xi^2 = {params.xi**2:.4f}")

    price, se = heston_mc_price_with_se(S0, K, T, r, params, n_paths=200_000)
    print(f"\nHeston MC Call Price: {price:.4f} (SE: {se:.4f})")

    # Degenerate test: xi=0, v0=theta=sigma^2 should match BS
    from models.black_scholes import bs_call_price

    sigma = 0.2
    params_bs = HestonParams(kappa=2.0, theta=sigma**2, xi=1e-8, rho=0.0, v0=sigma**2)
    heston_price = heston_mc_price(S0, K, T, r, params_bs, n_paths=500_000)
    bs_price_val = bs_call_price(S0, K, T, r, sigma)
    print(f"\n=== Degenerate Test (xi~0, v0=theta=sigma^2) ===")
    print(f"Heston MC: {heston_price:.4f}")
    print(f"BS:        {bs_price_val:.4f}")
    print(f"Diff:      {abs(heston_price - bs_price_val):.4f}")
