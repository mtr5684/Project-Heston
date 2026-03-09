# Option Pricing & Hedging under Stochastic Volatility (Heston Model)

## Motivation

The Black-Scholes model assumes constant volatility, yet market-observed option prices exhibit a **volatility smile** — implied volatility varies with strike and maturity. This systematic mispricing motivates stochastic volatility models.

The **Heston (1993)** model introduces a mean-reverting variance process correlated with the asset price, capturing the smile and leverage effect observed in equity markets. This project implements, calibrates, and applies the Heston model for option pricing and delta-hedging.

## Mathematical Models

### Black-Scholes

Under constant volatility σ, the stock follows geometric Brownian motion:

```
dS_t = r S_t dt + σ S_t dW_t
```

The closed-form European call price is:

```
C = S N(d₁) - K e^{-rT} N(d₂)
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Heston Stochastic Volatility

The Heston model introduces a stochastic variance process:

```
dS_t = r S_t dt + √v_t S_t dW_t¹
dv_t = κ(θ - v_t) dt + ξ √v_t dW_t²
Corr(dW¹, dW²) = ρ
```

**Parameters:**
| Symbol | Name | Typical Range |
|--------|------|---------------|
| κ | Mean-reversion speed | 0.1 – 10 |
| θ | Long-run variance | 0.01 – 1.0 |
| ξ | Vol-of-vol | 0.1 – 2.0 |
| ρ | Correlation | -0.99 – 0 |
| v₀ | Initial variance | 0.01 – 1.0 |

**Feller condition:** 2κθ > ξ² ensures variance stays strictly positive.

### Pricing Methods

1. **Monte Carlo (Euler-Maruyama):** Discretize the SDE system with reflection scheme for negative variance handling. Log-Euler for stock price to prevent negative values.

2. **Carr-Madan FFT:** Semi-analytical pricing using the Heston characteristic function:
   - Uses the "little Heston trap" formulation (Albrecher et al., 2007) for numerical stability
   - Simpson's rule quadrature weights
   - Single FFT call prices all strikes simultaneously (~1ms vs ~5s for MC)

## Project Structure

```
models/
├── black_scholes.py      # BS closed-form pricing + Monte Carlo
├── greeks.py             # Analytical BS Greeks + numerical bump-and-reprice
├── heston_mc.py          # Heston Monte Carlo simulation
└── heston_fourier.py     # Carr-Madan FFT pricing

calibration/
└── calibrate_heston.py   # Parameter calibration to market data

hedging/
└── delta_hedging.py      # Delta-hedging strategy + PnL simulation

experiments/
├── implied_vol_smile.py  # Market data + implied volatility surface
└── hedging_pnl.py        # Hedging experiment visualization

utils/
└── math_utils.py         # Shared numerical utilities
```

## Implementation

### Setup

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Stage 1: Black-Scholes pricing + MC convergence
python -m models.black_scholes

# Stage 1: Greeks computation
python -m models.greeks

# Stage 2: Fetch market data + implied volatility smile
python -m experiments.implied_vol_smile

# Stage 3: Heston MC pricing
python -m models.heston_mc

# Stage 3: Heston Fourier pricing + MC vs FFT comparison
python -m models.heston_fourier

# Stage 4: Calibrate Heston to market data
python -m calibration.calibrate_heston

# Stage 5: Delta-hedging PnL experiment
python -m experiments.hedging_pnl
```

All plots are saved to `results/`.

## Calibration

The calibration minimizes the **vega-weighted sum of squared implied volatility errors**:

```
L(Θ) = Σᵢ wᵢ (σ_model(Kᵢ, Tᵢ; Θ) - σ_market(Kᵢ, Tᵢ))²
```

where wᵢ = vegaᵢ / max(vega) normalizes the weights.

**Methodology:**
- Global optimizer: `scipy.optimize.differential_evolution`
- Prices computed via Carr-Madan FFT (one FFT per expiry, all strikes at once)
- Model IV obtained by inverting Heston prices through BS formula
- Parameter stability study with multi-start local optimization

## Hedging Results

The hedging experiment simulates a delta-hedging strategy:
- **True dynamics:** Heston model paths
- **BS hedge:** Uses constant-vol analytical delta
- **Heston hedge:** Uses bump-and-reprice numerical delta with Fourier pricer

**Key finding:** The Heston hedge produces a **tighter PnL distribution** (lower standard deviation) than the BS hedge, demonstrating the value of using the correct volatility model.

## Limitations

- **Euler-Maruyama bias:** Could be improved with QE scheme (Andersen, 2008)
- **Single-factor model:** No jumps, no rough volatility
- **Discrete rebalancing:** No transaction costs modeled
- **Calibration surface:** Non-convex, multiple local minima exist
- **Static calibration:** Parameters assumed constant over the hedging period

## Extensions

- **Bates model:** Heston + Merton jumps in the asset price
- **Rough Heston:** Replace Brownian variance driver with fractional Brownian motion
- **Transaction costs:** Incorporate bid-ask spread in hedging PnL
- **Variance reduction:** Antithetic variates, control variates for MC
- **QE discretization:** Quadratic-exponential scheme for better convergence

## References

- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility."
- Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform."
- Albrecher, H. et al. (2007). "The Little Heston Trap." (Numerical stability of characteristic function)
- Lord, R., Koekkoek, R., & van Dijk, D. (2010). "A Comparison of Biased Simulation Schemes for Stochastic Volatility Models."
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.*
