"""Hedging PnL experiment and visualization.

Orchestrates the hedging comparison between BS and Heston delta hedges
and produces publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from models.heston_mc import HestonParams
from hedging.delta_hedging import run_hedging_experiment
from utils.math_utils import ensure_results_dir


def plot_pnl_histogram(
    pnl_bs: np.ndarray,
    pnl_heston: np.ndarray,
    save_path: str = "results/hedging_pnl.png",
) -> None:
    """Overlaid histograms of BS hedge PnL and Heston hedge PnL."""
    ensure_results_dir("results")

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(pnl_bs.min(), pnl_heston.min()),
        max(pnl_bs.max(), pnl_heston.max()),
        50,
    )

    ax.hist(pnl_bs, bins=bins, alpha=0.5, label=f"BS hedge (std={np.std(pnl_bs):.3f})",
            color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.hist(pnl_heston, bins=bins, alpha=0.5, label=f"Heston hedge (std={np.std(pnl_heston):.3f})",
            color="tab:orange", edgecolor="black", linewidth=0.5)

    ax.axvline(np.mean(pnl_bs), color="tab:blue", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(pnl_heston), color="tab:orange", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Hedging PnL")
    ax.set_ylabel("Frequency")
    ax.set_title("Delta-Hedging PnL Distribution: BS vs Heston")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    summary = (
        f"BS:     mean={np.mean(pnl_bs):.3f}, std={np.std(pnl_bs):.3f}\n"
        f"Heston: mean={np.mean(pnl_heston):.3f}, std={np.std(pnl_heston):.3f}"
    )
    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PnL histogram saved to {save_path}")


def plot_pnl_summary_table(
    stats_bs: dict,
    stats_heston: dict,
    save_path: str = "results/hedging_summary.png",
) -> None:
    """Summary statistics table as a figure."""
    ensure_results_dir("results")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    headers = ["Metric", "BS Hedge", "Heston Hedge"]
    rows = []
    for key in ["mean", "std", "skew", "kurtosis", "min", "max"]:
        rows.append([key.capitalize(), f"{stats_bs[key]:.4f}", f"{stats_heston[key]:.4f}"])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("Hedging PnL Summary Statistics", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved to {save_path}")


def run_and_plot_hedging_comparison(
    params: HestonParams | None = None,
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma_bs: float = 0.2,
    n_paths: int = 1_000,
    n_steps: int = 252,
    rebalance_freq: int = 1,
    save_dir: str = "results",
) -> dict:
    """Main experiment: run hedging comparison and produce all plots."""
    if params is None:
        params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    ensure_results_dir(save_dir)

    print(f"Running hedging experiment: {n_paths} paths, {n_steps} steps...")
    print(f"  Heston params: kappa={params.kappa}, theta={params.theta}, "
          f"xi={params.xi}, rho={params.rho}, v0={params.v0}")
    print(f"  BS sigma: {sigma_bs}")

    result = run_hedging_experiment(
        S0, K, T, r, params, sigma_bs,
        n_paths=n_paths, n_steps=n_steps,
        rebalance_freq=rebalance_freq,
    )

    plot_pnl_histogram(
        result["pnl_bs"], result["pnl_heston"],
        save_path=f"{save_dir}/hedging_pnl.png",
    )

    plot_pnl_summary_table(
        result["stats_bs"], result["stats_heston"],
        save_path=f"{save_dir}/hedging_summary.png",
    )

    # Print results
    print("\n" + "=" * 50)
    print("HEDGING EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"\n{'Metric':<12} {'BS Hedge':>12} {'Heston Hedge':>14}")
    print("-" * 40)
    for key in ["mean", "std", "skew", "kurtosis"]:
        print(f"{key:<12} {result['stats_bs'][key]:>12.4f} {result['stats_heston'][key]:>14.4f}")

    ratio = result["stats_heston"]["std"] / result["stats_bs"]["std"]
    print(f"\nPnL Std Ratio (Heston/BS): {ratio:.2%}")
    if ratio < 1.0:
        print("=> Heston hedge produces TIGHTER PnL distribution")
    else:
        print("=> BS hedge produces tighter PnL (may need more paths/steps)")

    return result


if __name__ == "__main__":
    # Use moderate settings for a quick test
    run_and_plot_hedging_comparison(
        n_paths=500,
        n_steps=50,
        rebalance_freq=1,
    )
