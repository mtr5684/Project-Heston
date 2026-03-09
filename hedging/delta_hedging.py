"""Delta-hedging strategy and PnL simulation.

Compares BS delta hedge vs Heston numerical delta hedge
on paths simulated from the Heston model (the 'true' dynamics).
"""

import numpy as np

from models.heston_mc import HestonParams, simulate_heston
from models.heston_fourier import heston_fourier_price
from models.black_scholes import bs_call_price
from models.greeks import bs_delta, numerical_delta


def compute_hedge_delta(
    S: float,
    K: float,
    T_remaining: float,
    r: float,
    model: str = "bs",
    sigma: float | None = None,
    heston_params: HestonParams | None = None,
    bump: float = 0.01,
) -> float:
    """Compute delta for hedging.

    model="bs": analytical BS delta with given sigma.
    model="heston": numerical bump-and-reprice using heston_fourier_price.
    """
    if T_remaining <= 1e-6:
        # At expiry, delta is binary
        return 1.0 if S > K else 0.0

    if model == "bs":
        return bs_delta(S, K, T_remaining, r, sigma, "call")

    # Heston numerical delta
    def _pricing_fn(S0, K=K, T=T_remaining, r=r, params=heston_params):
        return heston_fourier_price(S0, K, T, r, params)

    return numerical_delta(_pricing_fn, S, bump=bump)


def simulate_hedge_pnl(
    S_path: np.ndarray,
    K: float,
    T: float,
    r: float,
    model: str = "bs",
    sigma: float | None = None,
    heston_params: HestonParams | None = None,
    rebalance_freq: int = 1,
) -> float:
    """Simulate hedging PnL along a single stock price path.

    Self-financing portfolio:
        - Short 1 call option
        - Long delta shares of stock
        - Cash account earns risk-free rate

    At each rebalancing date:
        1. Compute delta
        2. Rebalance (buy/sell shares)
        3. Update cash (borrow at rate r)

    Returns terminal PnL = portfolio value - option payoff.
    """
    n_steps = len(S_path) - 1
    dt = T / n_steps

    # Initial option price (premium received)
    if model == "bs":
        option_premium = bs_call_price(S_path[0], K, T, r, sigma)
    else:
        option_premium = heston_fourier_price(S_path[0], K, T, r, heston_params)

    shares = 0.0
    cash = option_premium  # premium received from selling the option

    for t in range(n_steps):
        T_remaining = T - t * dt

        if t % rebalance_freq == 0:
            new_delta = compute_hedge_delta(
                S_path[t], K, T_remaining, r,
                model=model, sigma=sigma,
                heston_params=heston_params,
            )
            # Rebalance: buy/sell shares
            cash -= (new_delta - shares) * S_path[t]
            shares = new_delta

        # Cash earns interest
        cash *= np.exp(r * dt)

    # At expiry: liquidate
    portfolio_value = shares * S_path[-1] + cash
    payoff = max(S_path[-1] - K, 0.0)

    return portfolio_value - payoff


def run_hedging_experiment(
    S0: float,
    K: float,
    T: float,
    r: float,
    heston_params: HestonParams,
    sigma_bs: float,
    n_paths: int = 1_000,
    n_steps: int = 252,
    rebalance_freq: int = 1,
    seed: int = 42,
) -> dict:
    """Full hedging experiment.

    1. Simulate n_paths Heston paths (the 'true' world)
    2. For each path, compute hedge PnL using BS delta
    3. For each path, compute hedge PnL using Heston delta
    4. Return statistics

    Returns dict with:
        'pnl_bs': array of PnLs under BS hedge
        'pnl_heston': array of PnLs under Heston hedge
        'stats_bs': dict with mean, std, skew, kurtosis
        'stats_heston': dict with mean, std, skew, kurtosis
    """
    from scipy.stats import skew, kurtosis

    # Simulate paths from true Heston dynamics
    S_paths, _ = simulate_heston(S0, r, heston_params, T, n_steps, n_paths, seed=seed)

    pnl_bs = np.zeros(n_paths)
    pnl_heston = np.zeros(n_paths)

    for i in range(n_paths):
        path = S_paths[i]

        pnl_bs[i] = simulate_hedge_pnl(
            path, K, T, r,
            model="bs", sigma=sigma_bs,
            rebalance_freq=rebalance_freq,
        )

        pnl_heston[i] = simulate_hedge_pnl(
            path, K, T, r,
            model="heston", heston_params=heston_params,
            rebalance_freq=rebalance_freq,
        )

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_paths} paths")

    def _stats(pnl):
        return {
            "mean": float(np.mean(pnl)),
            "std": float(np.std(pnl)),
            "skew": float(skew(pnl)),
            "kurtosis": float(kurtosis(pnl)),
            "min": float(np.min(pnl)),
            "max": float(np.max(pnl)),
        }

    return {
        "pnl_bs": pnl_bs,
        "pnl_heston": pnl_heston,
        "stats_bs": _stats(pnl_bs),
        "stats_heston": _stats(pnl_heston),
    }


if __name__ == "__main__":
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma_bs = 0.2

    print("Running hedging experiment (200 paths)...")
    result = run_hedging_experiment(
        S0, K, T, r, params, sigma_bs,
        n_paths=200, n_steps=50, rebalance_freq=1,
    )

    print(f"\n=== BS Hedge PnL Stats ===")
    for k, v in result["stats_bs"].items():
        print(f"  {k}: {v:.4f}")

    print(f"\n=== Heston Hedge PnL Stats ===")
    for k, v in result["stats_heston"].items():
        print(f"  {k}: {v:.4f}")

    print(f"\nHeston std / BS std = {result['stats_heston']['std'] / result['stats_bs']['std']:.2%}")
