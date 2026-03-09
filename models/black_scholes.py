"""Black-Scholes pricing model: closed-form and Monte Carlo."""

import numpy as np
import matplotlib.pyplot as plt
from utils.math_utils import norm_cdf, norm_pdf, ensure_results_dir


def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 = d1 - sigma * sqrt(T)."""
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price (closed-form).

    C = S * N(d1) - K * exp(-rT) * N(d2)
    """
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price via put-call parity.

    P = K * exp(-rT) * N(-d2) - S * N(-d1)
    """
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Black-Scholes price for call or put."""
    if option_type == "call":
        return bs_call_price(S, K, T, r, sigma)
    return bs_put_price(S, K, T, r, sigma)


def bs_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 100_000,
    option_type: str = "call",
    seed: int = 42,
) -> float:
    """Monte Carlo price under GBM using exact log-normal terminal distribution.

    S_T = S * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    return np.exp(-r * T) * np.mean(payoffs)


def bs_mc_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    sim_counts: list[int] | None = None,
    seed: int = 42,
) -> tuple[list[int], list[float], list[float]]:
    """Run MC pricing for increasing simulation counts.

    Returns (sim_counts, mc_prices, absolute_errors_vs_closed_form).
    """
    if sim_counts is None:
        sim_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    closed_form = bs_call_price(S, K, T, r, sigma)
    mc_prices = []
    errors = []

    for n in sim_counts:
        price = bs_mc_price(S, K, T, r, sigma, n_sims=n, seed=seed)
        mc_prices.append(price)
        errors.append(abs(price - closed_form))

    return sim_counts, mc_prices, errors


def plot_mc_convergence(
    sim_counts: list[int],
    mc_prices: list[float],
    closed_form: float,
    save_path: str | None = None,
) -> None:
    """Plot MC price convergence towards the Black-Scholes closed-form price."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MC price vs closed-form
    ax1.semilogx(sim_counts, mc_prices, "o-", label="MC Price")
    ax1.axhline(y=closed_form, color="r", linestyle="--", label=f"BS = {closed_form:.4f}")
    ax1.set_xlabel("Number of simulations")
    ax1.set_ylabel("Option price")
    ax1.set_title("Monte Carlo Convergence to Black-Scholes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: absolute error
    errors = [abs(p - closed_form) for p in mc_prices]
    ax2.loglog(sim_counts, errors, "s-", color="tab:orange", label="|MC - BS|")
    # Reference O(1/sqrt(N)) line
    ref = errors[0] * np.sqrt(sim_counts[0]) / np.sqrt(np.array(sim_counts))
    ax2.loglog(sim_counts, ref, "--", color="gray", alpha=0.6, label=r"$O(1/\sqrt{N})$")
    ax2.set_xlabel("Number of simulations")
    ax2.set_ylabel("Absolute error")
    ax2.set_title("MC Error Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        ensure_results_dir(str(save_path).rsplit("/", 1)[0] if "/" in str(save_path) else "results")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)
    mc = bs_mc_price(S, K, T, r, sigma, n_sims=500_000)

    print(f"BS Call Price:  {call:.4f}")
    print(f"BS Put Price:   {put:.4f}")
    print(f"MC Call Price:  {mc:.4f}")
    print(f"Put-Call Parity check: C - P = {call - put:.4f}, S - Ke^(-rT) = {S - K * np.exp(-r * T):.4f}")

    counts, prices, errors = bs_mc_convergence(S, K, T, r, sigma)
    plot_mc_convergence(counts, prices, call, save_path="results/mc_convergence.png")
    print("Convergence plot saved to results/mc_convergence.png")
