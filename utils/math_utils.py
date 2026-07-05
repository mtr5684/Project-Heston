"""Shared mathematical utilities for the Heston project."""

import os

import numpy as np
from scipy.stats import norm


def norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal cumulative distribution function N(x)."""
    return norm.cdf(x)


def norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal probability density function n(x)."""
    return norm.pdf(x)


def generate_correlated_bm(
    n_paths: int,
    n_steps: int,
    dt: float,
    rho: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated Brownian motion increment matrices.

    Uses Cholesky decomposition of the correlation matrix:
        dW1 = Z1 * sqrt(dt)
        dW2 = (rho * Z1 + sqrt(1 - rho^2) * Z2) * sqrt(dt)

    Returns:
        (dW1, dW2): each of shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))

    dW1 = Z1 * np.sqrt(dt)
    dW2 = (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2) * np.sqrt(dt)

    return dW1, dW2


def ensure_results_dir(path: str = "results") -> str:
    """Create results directory if it does not exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path
