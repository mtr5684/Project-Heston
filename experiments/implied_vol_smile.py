"""Implied volatility computation and volatility smile visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from datetime import datetime

from models.black_scholes import bs_price
from models.greeks import bs_vega
from utils.math_utils import ensure_results_dir


# ---------------------------------------------------------------------------
# Implied Volatility Solvers
# ---------------------------------------------------------------------------

def implied_vol_brentq(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Brentq root-finding for IV: solve BS(sigma) - market_price = 0 on [0.001, 5.0].

    More robust than Newton but slower.
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    return brentq(objective, 1e-4, 5.0, xtol=1e-10)


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    sigma0: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver.

    sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

    Falls back to Brentq if Newton fails to converge.
    """
    sigma = sigma0
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        if sigma <= 0:
            break

    # Fallback to Brentq
    return implied_vol_brentq(market_price, S, K, T, r, option_type)


# ---------------------------------------------------------------------------
# Market Data Fetching
# ---------------------------------------------------------------------------

def fetch_option_chain(
    ticker: str = "SPY",
    save_path: str = "data/options_chain.csv",
) -> pd.DataFrame:
    """Fetch option chain from yfinance.

    Selects 2-3 expiration dates (near, medium, far).
    Computes mid-price = (bid + ask) / 2.
    Saves to CSV and returns DataFrame.
    """
    import yfinance as yf

    stock = yf.Ticker(ticker)
    S = stock.history(period="1d")["Close"].iloc[-1]

    expirations = stock.options
    # Pick near (1-2 months), medium (3-6 months), far (6-12 months)
    today = datetime.now()
    selected = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        days = (exp_date - today).days
        if 20 <= days <= 365:
            selected.append(exp_str)
        if len(selected) >= 3:
            break

    if not selected:
        selected = list(expirations[:3])

    rows = []
    for exp_str in selected:
        chain = stock.option_chain(exp_str)
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                mid = (row.get("bid", 0) + row.get("ask", 0)) / 2.0
                if mid <= 0 or row.get("volume", 0) is None:
                    continue
                vol = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                if vol < 5 or oi < 5:
                    continue

                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                T = max((exp_date - today).days / 365.0, 1 / 365)

                rows.append({
                    "strike": row["strike"],
                    "expiry": exp_str,
                    "mid_price": mid,
                    "option_type": opt_type,
                    "bid": row.get("bid", 0),
                    "ask": row.get("ask", 0),
                    "volume": vol,
                    "openInterest": oi,
                    "yf_impliedVol": row.get("impliedVolatility", np.nan),
                    "underlyingPrice": S,
                    "T": T,
                    "r": 0.045,  # approximate risk-free rate
                })

    df = pd.DataFrame(rows)

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} options to {save_path} (S={S:.2f})")
    return df


# ---------------------------------------------------------------------------
# IV Surface Computation
# ---------------------------------------------------------------------------

def compute_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied volatility for each row in the option chain DataFrame.

    Adds 'iv_computed' column. Drops rows where IV computation fails.
    """
    ivs = []
    for _, row in df.iterrows():
        try:
            iv = implied_vol_newton(
                market_price=row["mid_price"],
                S=row["underlyingPrice"],
                K=row["strike"],
                T=row["T"],
                r=row["r"],
                option_type=row["option_type"],
            )
            if 0.01 < iv < 3.0:
                ivs.append(iv)
            else:
                ivs.append(np.nan)
        except Exception:
            ivs.append(np.nan)

    df = df.copy()
    df["iv_computed"] = ivs
    df = df.dropna(subset=["iv_computed"])
    return df


# ---------------------------------------------------------------------------
# Volatility Smile Plot
# ---------------------------------------------------------------------------

def plot_vol_smile(
    df: pd.DataFrame,
    save_path: str = "results/vol_smile.png",
) -> None:
    """Plot IV vs moneyness (K/S) grouped by expiry.

    Demonstrates that BS constant-vol assumption is violated in practice.
    """
    ensure_results_dir("results")

    df = df.copy()
    df["moneyness"] = df["strike"] / df["underlyingPrice"]

    fig, ax = plt.subplots(figsize=(10, 6))

    calls = df[df["option_type"] == "call"]
    for expiry, group in calls.groupby("expiry"):
        group_sorted = group.sort_values("moneyness")
        ax.plot(
            group_sorted["moneyness"],
            group_sorted["iv_computed"] * 100,
            "o-",
            markersize=4,
            label=f"T = {expiry}",
        )

    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("Volatility Smile from Market Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Commentary
    ax.text(
        0.02, 0.02,
        "The volatility smile shows that Black-Scholes constant-vol\n"
        "assumption is violated: OTM puts trade at higher implied vol (skew).",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        fontstyle="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Volatility smile plot saved to {save_path}")


if __name__ == "__main__":
    print("Fetching option chain...")
    df = fetch_option_chain("SPY")

    print("Computing implied volatilities...")
    df = compute_iv_surface(df)
    print(f"Successfully computed IV for {len(df)} options")

    plot_vol_smile(df)
