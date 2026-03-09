"""Heston model pricing via characteristic function and Carr-Madan FFT.

Uses the 'little Heston trap' formulation (Albrecher et al., 2007)
for numerical stability of the characteristic function.
"""

import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from models.heston_mc import HestonParams, heston_mc_price_with_se


def heston_characteristic_function(
    u: np.ndarray | complex,
    T: float,
    r: float,
    params: HestonParams,
    S0: float,
) -> np.ndarray | complex:
    """Heston characteristic function for log-price X_T = ln(S_T).

    phi(u) = E[exp(i*u*X_T)]

    Uses the 'little Heston trap' formulation to avoid branch-cut
    discontinuities in the complex logarithm.

    Reference: Albrecher, Mayer, Schoutens, Tistaert (2007)
    """
    kappa, theta, xi, rho, v0 = (
        params.kappa, params.theta, params.xi, params.rho, params.v0
    )

    # Forward log-price
    x = np.log(S0) + r * T

    # Ensure u is array for uniform handling
    scalar_input = np.isscalar(u) or (isinstance(u, np.ndarray) and u.ndim == 0)
    u = np.atleast_1d(np.asarray(u, dtype=complex))

    # Complex-valued intermediate quantities
    iu = 1j * u
    d = np.sqrt(
        (rho * xi * iu - kappa) ** 2 + xi**2 * (iu + u**2)
    )

    # "Little Heston trap" formulation
    # g = (kappa - rho*xi*iu + d) / (kappa - rho*xi*iu - d)
    numer = kappa - rho * xi * iu + d
    denom = kappa - rho * xi * iu - d

    # Handle u=0 case: d=kappa, numer=2*kappa, denom=0 -> use L'Hopital / direct formula
    # At u=0: phi(0) = 1 by definition (characteristic function at 0)
    safe = np.abs(denom) > 1e-14
    denom_safe = np.where(safe, denom, 1.0)  # avoid division by zero
    g = np.where(safe, numer / denom_safe, 0.0)

    # C and D functions
    exp_dT = np.exp(d * T)

    # For the log term: log((g*exp(dT) - 1) / (g - 1))
    # When g is huge (u~0), this simplifies to log(exp(dT)) = dT
    log_arg_num = g * exp_dT - 1.0
    log_arg_den = g - 1.0
    log_ratio = np.where(
        safe,
        np.log(log_arg_num / np.where(np.abs(log_arg_den) > 1e-14, log_arg_den, 1.0)),
        d * T,
    )

    C = (kappa * theta / xi**2) * (numer * T - 2.0 * log_ratio)

    D_numer = 1.0 - exp_dT
    D_denom = 1.0 - g * exp_dT
    D = np.where(
        safe,
        (numer / xi**2) * D_numer / np.where(np.abs(D_denom) > 1e-14, D_denom, 1.0),
        0.0,
    )

    result = np.exp(C + D * v0 + iu * x)

    if scalar_input:
        return result[0]
    return result


def carr_madan_fft(
    S0: float,
    r: float,
    T: float,
    params: HestonParams,
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Carr-Madan FFT pricing for European call options.

    The modified call transform (damped by exp(-alpha*k)):
        psi(v) = exp(-rT) * phi(v - (alpha+1)*i)
                 / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)

    FFT converts the integral into a sum over N points with
    Simpson's rule weights for improved quadrature accuracy.

    Returns:
        strikes: array of strike prices (length N)
        call_prices: corresponding call prices (length N)
    """
    # Log-strike spacing: lambda = 2*pi / (N*eta)
    lam = 2.0 * np.pi / (N * eta)

    # Integration grid in u-space
    j = np.arange(N)
    v = eta * j

    # Log-strike grid
    b = N * lam / 2.0  # centering
    k = -b + lam * j

    # Modified characteristic function (Carr-Madan integrand)
    iu_alpha = 1j * (alpha + 1)
    phi = heston_characteristic_function(v - iu_alpha, T, r, params, S0)

    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    psi = np.exp(-r * T) * phi / denom

    # Simpson's rule weights
    simpson = 3.0 + (-1.0) ** (j + 1)
    simpson[0] = 1.0
    simpson = simpson / 3.0

    # FFT input
    x = np.exp(1j * v * (b)) * psi * eta * simpson

    # Perform FFT
    fft_result = np.fft.fft(x)

    # Extract call prices
    call_prices = (np.exp(-alpha * k) / np.pi) * np.real(fft_result)

    # Convert log-strikes to strikes
    strikes = np.exp(k)

    return strikes, call_prices


def heston_fourier_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str = "call",
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """Price a single European option under Heston using Carr-Madan FFT.

    Runs the full FFT to get a spectrum of call prices, then interpolates
    to the desired strike K.

    For puts: use put-call parity P = C - S0*exp(-qT) + K*exp(-rT).
    (No dividend yield here, so q=0.)
    """
    strikes, call_prices = carr_madan_fft(S0, r, T, params, N, alpha, eta)

    # Filter to positive prices and reasonable strike range
    valid = (strikes > 0) & (call_prices > 0) & (strikes < 5 * S0)
    if not np.any(valid):
        return np.nan

    interp_fn = interp1d(
        np.log(strikes[valid]),
        call_prices[valid],
        kind="cubic",
        fill_value="extrapolate",
    )
    call_price = float(interp_fn(np.log(K)))
    call_price = max(call_price, 0.0)

    if option_type == "call":
        return call_price
    # Put-call parity
    return call_price - S0 + K * np.exp(-r * T)


def heston_fourier_price_multi_strike(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str = "call",
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> np.ndarray:
    """Price multiple strikes at once (single FFT call + interpolation).

    This is the workhorse for calibration: one FFT gives all strikes.
    """
    fft_strikes, call_prices = carr_madan_fft(S0, r, T, params, N, alpha, eta)

    valid = (fft_strikes > 0) & (call_prices > 0) & (fft_strikes < 5 * S0)
    if not np.any(valid):
        return np.full_like(strikes, np.nan, dtype=float)

    interp_fn = interp1d(
        np.log(fft_strikes[valid]),
        call_prices[valid],
        kind="cubic",
        fill_value=np.nan,
        bounds_error=False,
    )
    prices = interp_fn(np.log(strikes))
    prices = np.maximum(prices, 0.0)

    if option_type == "put":
        prices = prices - S0 + strikes * np.exp(-r * T)

    return prices


def compare_mc_vs_fourier(
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    mc_paths_list: list[int] | None = None,
) -> pd.DataFrame:
    """Compare MC and Fourier prices for given parameters.

    Returns DataFrame with timing and accuracy comparison.
    """
    if mc_paths_list is None:
        mc_paths_list = [10_000, 50_000, 100_000]

    # Fourier price (reference)
    t0 = time.time()
    fourier_price = heston_fourier_price(S0, K, T, r, params)
    fourier_time = time.time() - t0

    results = []
    for n_paths in mc_paths_list:
        t0 = time.time()
        mc_price, mc_se = heston_mc_price_with_se(
            S0, K, T, r, params, n_paths=n_paths
        )
        mc_time = time.time() - t0

        results.append({
            "n_paths": n_paths,
            "mc_price": mc_price,
            "mc_se": mc_se,
            "fourier_price": fourier_price,
            "mc_time_s": mc_time,
            "fourier_time_s": fourier_time,
            "error_mc_vs_fourier": abs(mc_price - fourier_price),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    # Sanity checks on characteristic function
    phi_0 = heston_characteristic_function(0.0, T, r, params, S0)
    phi_neg_i = heston_characteristic_function(-1j, T, r, params, S0)
    forward = S0 * np.exp(r * T)

    print("=== Characteristic Function Sanity ===")
    print(f"phi(0) = {phi_0:.6f}  (should be 1.0)")
    print(f"phi(-i) = {np.real(phi_neg_i):.4f}  (should be forward = {forward:.4f})")

    # Single price
    fourier_price = heston_fourier_price(S0, K, T, r, params)
    print(f"\nHeston Fourier Call Price: {fourier_price:.4f}")

    # Multi-strike
    strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
    prices = heston_fourier_price_multi_strike(S0, strikes, T, r, params)
    print("\n=== Multi-Strike Pricing ===")
    for k, p in zip(strikes, prices):
        print(f"  K={k:6.0f}  Call={p:.4f}")

    # MC vs Fourier comparison
    print("\n=== MC vs Fourier Comparison ===")
    df = compare_mc_vs_fourier(S0, K, T, r, params)
    print(df.to_string(index=False))
