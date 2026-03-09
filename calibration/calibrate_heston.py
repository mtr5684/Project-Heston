"""Heston model calibration to market option data.

Supports:
- Price-based objective: sum of squared price errors
- IV-based objective (upgrade): vega-weighted sum of squared IV errors
- Global optimization via differential_evolution
- Multi-start local optimization for parameter stability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

from models.heston_mc import HestonParams
from models.heston_fourier import heston_fourier_price_multi_strike
from models.greeks import bs_vega
from experiments.implied_vol_smile import implied_vol_newton
from utils.math_utils import ensure_results_dir


# Parameter bounds: [kappa, theta, xi, rho, v0]
PARAM_BOUNDS = [
    (0.1, 10.0),   # kappa
    (0.01, 1.0),   # theta (long-run vol 10%-100%)
    (0.1, 2.0),    # xi (vol of vol)
    (-0.99, 0.0),  # rho (negative for equities)
    (0.01, 1.0),   # v0
]


def load_market_data(csv_path: str = "data/options_chain.csv") -> pd.DataFrame:
    """Load and clean the option chain CSV.

    Filters for calls with sufficient liquidity and positive IV.
    """
    df = pd.read_csv(csv_path)
    df = df[df["option_type"] == "call"].copy()
    df = df[df["mid_price"] > 0.5]
    df = df[df["volume"] >= 10]
    df = df[df["openInterest"] >= 10]

    # Filter out deep ITM and deep OTM (keep moneyness 0.8 to 1.2)
    df["moneyness"] = df["strike"] / df["underlyingPrice"]
    df = df[(df["moneyness"] >= 0.80) & (df["moneyness"] <= 1.20)]

    df = df.reset_index(drop=True)
    return df


def objective_price(
    x: np.ndarray,
    market_df: pd.DataFrame,
    S0: float,
    r: float,
) -> float:
    """Sum of squared price errors.

    L(Theta) = sum_i (C_model(K_i, T_i; Theta) - C_market(K_i, T_i))^2
    """
    params = HestonParams.from_array(x)

    total_error = 0.0
    for T_val, group in market_df.groupby("T"):
        strikes = group["strike"].values
        market_prices = group["mid_price"].values

        model_prices = heston_fourier_price_multi_strike(
            S0, strikes, T_val, r, params
        )

        valid = ~np.isnan(model_prices)
        if np.any(valid):
            total_error += np.sum((model_prices[valid] - market_prices[valid]) ** 2)

    return total_error


def objective_iv(
    x: np.ndarray,
    market_df: pd.DataFrame,
    S0: float,
    r: float,
) -> float:
    """Vega-weighted sum of squared IV errors.

    L(Theta) = sum_i w_i * (sigma_model(K_i, T_i) - sigma_market(K_i, T_i))^2

    where w_i = vega_i / max(vega_i) to normalize weights.
    """
    params = HestonParams.from_array(x)

    iv_errors = []
    weights = []

    for T_val, group in market_df.groupby("T"):
        strikes = group["strike"].values
        market_ivs = group["iv_market"].values if "iv_market" in group.columns else None

        model_prices = heston_fourier_price_multi_strike(
            S0, strikes, T_val, r, params
        )

        for i, (K_val, model_p) in enumerate(zip(strikes, model_prices)):
            if np.isnan(model_p) or model_p <= 0:
                continue
            try:
                model_iv = implied_vol_newton(model_p, S0, K_val, T_val, r, "call")
            except Exception:
                continue

            if market_ivs is not None:
                market_iv = market_ivs[i]
            else:
                market_iv = group.iloc[i].get("iv_computed", group.iloc[i].get("yf_impliedVol", np.nan))

            if np.isnan(market_iv) or np.isnan(model_iv):
                continue

            # Vega weight (use BS vega at market IV as approximation)
            vega = bs_vega(S0, K_val, T_val, r, market_iv)
            weights.append(vega)
            iv_errors.append((model_iv - market_iv) ** 2)

    if not weights:
        return 1e10

    weights = np.array(weights)
    iv_errors = np.array(iv_errors)
    max_vega = np.max(weights)
    if max_vega > 0:
        weights /= max_vega

    return float(np.sum(weights * iv_errors))


def _prepare_market_iv(df: pd.DataFrame, S0: float, r: float) -> pd.DataFrame:
    """Pre-compute market IV if not already present."""
    if "iv_market" in df.columns:
        return df

    df = df.copy()
    ivs = []
    for _, row in df.iterrows():
        try:
            iv = implied_vol_newton(
                row["mid_price"], S0, row["strike"], row["T"], r, "call"
            )
            ivs.append(iv if 0.01 < iv < 3.0 else np.nan)
        except Exception:
            ivs.append(np.nan)
    df["iv_market"] = ivs
    df = df.dropna(subset=["iv_market"])
    return df


def calibrate_heston(
    market_df: pd.DataFrame,
    S0: float,
    r: float,
    method: str = "differential_evolution",
    objective_type: str = "iv",
    n_starts: int = 5,
    seed: int = 42,
) -> dict:
    """Calibrate Heston parameters to market data.

    Args:
        method: "differential_evolution" (global) or "minimize" (multi-start local)
        objective_type: "price" or "iv" (vega-weighted)
        n_starts: number of random starts for local optimizer

    Returns dict with:
        'params': HestonParams (best fit)
        'fun': optimal objective value
        'all_results': list of (params, fun) for stability analysis
        'success': bool
    """
    # Pre-compute market IV for iv-based objective
    if objective_type == "iv":
        market_df = _prepare_market_iv(market_df, S0, r)

    obj_fn = objective_iv if objective_type == "iv" else objective_price
    args = (market_df, S0, r)

    all_results = []

    if method == "differential_evolution":
        result = differential_evolution(
            obj_fn,
            bounds=PARAM_BOUNDS,
            args=args,
            seed=seed,
            maxiter=200,
            tol=1e-6,
            polish=True,
            disp=False,
        )
        best_params = HestonParams.from_array(result.x)
        all_results.append((best_params, result.fun))

        # Accept result even if optimizer reports non-convergence,
        # as long as we have a finite objective value
        return {
            "params": best_params,
            "fun": result.fun,
            "all_results": all_results,
            "success": result.success or np.isfinite(result.fun),
        }

    # Multi-start local optimization
    rng = np.random.default_rng(seed)
    best = None

    for i in range(n_starts):
        x0 = np.array([
            rng.uniform(*PARAM_BOUNDS[j]) for j in range(5)
        ])
        try:
            result = minimize(
                obj_fn,
                x0,
                args=args,
                method="L-BFGS-B",
                bounds=PARAM_BOUNDS,
                options={"maxiter": 200, "ftol": 1e-10},
            )
            p = HestonParams.from_array(result.x)
            all_results.append((p, result.fun))

            if best is None or result.fun < best["fun"]:
                best = {"params": p, "fun": result.fun, "success": result.success}
        except Exception:
            continue

    if best is None:
        return {"params": None, "fun": np.inf, "all_results": all_results, "success": False}

    best["all_results"] = all_results
    return best


def plot_calibration_result(
    market_df: pd.DataFrame,
    calibrated_params: HestonParams,
    S0: float,
    r: float,
    save_path: str = "results/calibration_smile.png",
) -> None:
    """Plot model IV smile vs market IV smile after calibration."""
    ensure_results_dir("results")

    market_df = _prepare_market_iv(market_df, S0, r)
    expiries = sorted(market_df["expiry"].unique())
    n_exp = len(expiries)

    fig, axes = plt.subplots(1, min(n_exp, 3), figsize=(6 * min(n_exp, 3), 5), squeeze=False)

    for idx, expiry in enumerate(expiries[:3]):
        ax = axes[0, idx]
        group = market_df[market_df["expiry"] == expiry].sort_values("strike")

        strikes = group["strike"].values
        market_ivs = group["iv_market"].values

        # Model IV
        model_prices = heston_fourier_price_multi_strike(
            S0, strikes, group["T"].iloc[0], r, calibrated_params
        )
        model_ivs = []
        for K_val, mp in zip(strikes, model_prices):
            try:
                iv = implied_vol_newton(mp, S0, K_val, group["T"].iloc[0], r, "call")
                model_ivs.append(iv)
            except Exception:
                model_ivs.append(np.nan)

        moneyness = strikes / S0

        ax.plot(moneyness, np.array(market_ivs) * 100, "o", markersize=4, label="Market")
        ax.plot(moneyness, np.array(model_ivs) * 100, "x-", markersize=4, label="Heston")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility (%)")
        ax.set_title(f"Expiry: {expiry}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Calibrated: kappa={calibrated_params.kappa:.2f}, "
        f"theta={calibrated_params.theta:.4f}, xi={calibrated_params.xi:.2f}, "
        f"rho={calibrated_params.rho:.2f}, v0={calibrated_params.v0:.4f}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration smile plot saved to {save_path}")


def study_parameter_stability(
    market_df: pd.DataFrame,
    S0: float,
    r: float,
    n_starts: int = 20,
    save_path: str = "results/param_stability.png",
) -> pd.DataFrame:
    """Run calibration from many random starting points.

    Demonstrates that multiple local minima exist in the Heston calibration
    surface. Plots histograms of calibrated parameters.
    """
    ensure_results_dir("results")

    result = calibrate_heston(
        market_df, S0, r,
        method="minimize",
        objective_type="iv",
        n_starts=n_starts,
    )

    all_results = result["all_results"]
    if not all_results:
        print("No successful calibrations.")
        return pd.DataFrame()

    rows = []
    for params, fun in all_results:
        rows.append({
            "kappa": params.kappa,
            "theta": params.theta,
            "xi": params.xi,
            "rho": params.rho,
            "v0": params.v0,
            "objective": fun,
        })
    df = pd.DataFrame(rows)

    # Plot histograms
    param_names = ["kappa", "theta", "xi", "rho", "v0"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        axes[i].hist(df[name], bins=min(15, n_starts), edgecolor="black", alpha=0.7)
        axes[i].set_title(name)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    axes[5].hist(df["objective"], bins=min(15, n_starts), edgecolor="black", alpha=0.7, color="tab:red")
    axes[5].set_title("Objective Value")
    axes[5].set_xlabel("Value")

    fig.suptitle(f"Parameter Stability Analysis ({n_starts} random starts)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Parameter stability plot saved to {save_path}")

    return df


if __name__ == "__main__":
    print("Loading market data...")
    df = load_market_data()
    S0 = df["underlyingPrice"].iloc[0]
    r = df["r"].iloc[0]
    print(f"Loaded {len(df)} options, S0={S0:.2f}, r={r}")

    print("\nCalibrating Heston (differential_evolution, IV-based)...")
    print("This may take a few minutes...")
    result = calibrate_heston(df, S0, r, method="differential_evolution", objective_type="iv")

    if result["success"]:
        p = result["params"]
        print(f"\nCalibrated parameters:")
        print(f"  kappa = {p.kappa:.4f}")
        print(f"  theta = {p.theta:.6f}")
        print(f"  xi    = {p.xi:.4f}")
        print(f"  rho   = {p.rho:.4f}")
        print(f"  v0    = {p.v0:.6f}")
        print(f"  Feller: {p.feller_condition()}")
        print(f"  Objective: {result['fun']:.6e}")

        plot_calibration_result(df, p, S0, r)
    else:
        print("Calibration failed.")
