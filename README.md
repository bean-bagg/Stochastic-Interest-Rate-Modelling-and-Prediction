# Stochastic Interest Rate Modelling and Prediction

**Objective:** Implementing, calibrating, and extending the Cox-Ingersoll-Ross (CIR) model on real yield curve data.

---

## Table of Contents
1. [Mathematical Setup](#mathematical-setup)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
   - [A. Data Engineering & Preprocessing](#a-data-engineering--preprocessing)
   - [B. Model Calibration](#b-model-calibration)
   - [C. Prediction Challenge](#c-prediction-challenge-reconstruct-the-curve-from-only-3m)
   - [D. Extensions](#d-extensions)
5. [Results](#results)
   - [Calibrated Parameters](#calibrated-parameters)
   - [Inter-Tenor Correlation](#inter-tenor-correlation)
   - [Base CIR — Per-Tenor Test Metrics](#base-cir--per-tenor-test-metrics)
   - [Full Model Comparison](#full-model-comparison)
   - [3M-Only Test Predictions (Sample)](#3m-only-test-predictions-sample)
6. [Key Inferences](#key-inferences)
7. [Dependencies](#dependencies)

---

## Mathematical Setup

The CIR short-rate process is:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

For a zero-coupon bond with maturity $\tau = T - t$:

$$P(t,T) = A(\tau)\,e^{-B(\tau)r_t}, \qquad y(t,\tau) = \frac{B(\tau)r_t - \log A(\tau)}{\tau}$$

The observed **3-month (3M) zero-coupon yield** is used as a proxy for the latent instantaneous short rate $r_t$. This is a deliberate restriction: a single factor can mostly capture the **level** of the curve, but it cannot independently control slope and curvature. The extension section therefore tests an empirical CIR++ residual correction and a two-factor level-slope benchmark to address this limitation.

---

## Project Structure

```
Stochastic-Interest-Rate-Modelling-and-Prediction/
│
├── CIR_SDE.ipynb          # Main notebook
├── train_data.csv         # Training data (2016-05-19 to 2024-04-26)
├── test_data.csv          # Full test data with all tenors (2024-04-29 to 2026-04-29)
├── test_data_3M.csv       # Test data with 3M yield only (for prediction challenge)
├── predicted_yield_       # Predicted Yield Curve
|  curve_from_test_3M.csv
└── README.md
```

---

## Dataset

The dataset contains daily **zero-coupon yield curve** data across 9 tenors:

| Column | Maturity |
|--------|----------|
| ZC025YR | 0.25 yr (3M) |
| ZC050YR | 0.50 yr (6M) |
| ZC075YR | 0.75 yr (9M) |
| ZC100YR | 1.00 yr (1Y) |
| ZC200YR | 2.00 yr (2Y) |
| ZC500YR | 5.00 yr (5Y) |
| ZC1000YR | 10.00 yr (10Y) |
| ZC2000YR | 20.00 yr (20Y) |
| ZC3000YR | 30.00 yr (30Y) |

**Split summary:**

| Split | Rows | Date Range |
|-------|------|------------|
| Train | 1,976 | 2016-05-19 → 2024-04-26 |
| Test (full) | 495 | 2024-04-29 → 2026-04-29 |
| Test (3M only) | 495 | 2024-04-29 → 2026-04-29 |

**Training set descriptive statistics (yields as decimals):**

| Tenor | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| ZC025YR | 0.016695 | 0.016641 | 0.000486 | 0.051962 |
| ZC050YR | 0.017885 | 0.016761 | 0.000878 | 0.053195 |
| ZC075YR | 0.018530 | 0.016649 | 0.001054 | 0.054040 |
| ZC100YR | 0.019174 | 0.016587 | 0.001227 | 0.054941 |
| ZC200YR | 0.018063 | 0.013661 | 0.001417 | 0.048496 |
| ZC500YR | 0.018109 | 0.010396 | 0.002786 | 0.043147 |
| ZC1000YR | 0.020226 | 0.008805 | 0.004451 | 0.042232 |
| ZC2000YR | 0.022823 | 0.007136 | 0.008394 | 0.040687 |
| ZC3000YR | 0.022619 | 0.006601 | 0.006921 | 0.039306 |

---

## Methodology

### A. Data Engineering & Preprocessing

Raw yields are cleaned using a **robust local outlier filter**:
- A rolling 63-day median and MAD (median absolute deviation) are computed for each tenor column.
- Observations with a robust Z-score exceeding 8 are flagged as outliers and set to `NaN`.
- Missing values are filled via time-based interpolation, then forward/backward filled.
- All yields are clipped at a floor of `1e-6` to prevent negative inputs to the CIR square-root term.

### B. Model Calibration

Two calibration methods are implemented and compared:

**1. OLS Short-Rate Dynamics**
Treats the discrete 3M series as an AR(1) process:
$$r_{t+\Delta t} = a + b \cdot r_t + \varepsilon$$
Parameters are recovered as:
- $\kappa = -\log(b)/\Delta t$
- $\theta = a/(1-b)$
- $\sigma$ estimated from scaled residual variance

This is useful for interpreting **shock persistence** (higher $\kappa$ → faster mean reversion) but optimises for one-step short-rate forecasting rather than curve shape.

**2. Cross-Sectional Least Squares** *(used for prediction)*
Calibrates $(\kappa, \theta, \sigma)$ by minimising the mean squared error between CIR-implied zero-coupon yields and observed yields **across all training dates and tenors simultaneously**, using the 3M rate as the state variable. A random search over 25,000 candidate parameter sets is followed by Nelder-Mead refinement. A small penalty term discourages — but does not prohibit — Feller condition violations:

$$\text{loss} = \text{MSE} + 10^{-4} \cdot \max\!\left(0,\; \sigma^2 - 2\kappa\theta\right)^2$$

### C. Prediction Challenge: Reconstruct the Curve from Only 3M

Given **only the 3M yield** from the test set, the model reconstructs the yields at 6M, 9M, 1Y, and 2Y using the calibrated CIR zero-coupon yield formula. This simulates a real-world scenario where only one observable point on the curve is available.

### D. Extensions

Three extensions are implemented and benchmarked against the base CIR model:

**1. CIR++ Residual Correction**
Learns a degree-3 polynomial of the 3M rate to correct the base CIR residuals across all tenors via ridge regression. Motivated by the CIR++ framework of deterministic shifts.

**2. Selective Rate-Time CIR++** *(best performing)*
Keeps the base CIR predictions for the 6M/9M/1Y sector unchanged and applies a deterministic, **recency-weighted** correction only to the 2Y+ tenor. Features are: normalised 3M rate, normalised time index, and their interaction. Exponential decay weighting (half-life = 504 trading days ≈ 2 years) emphasises recent data.

**3. Empirical Level-Slope Benchmark**
A two-factor empirical model where the level factor is the observed 3M rate and the slope is inferred via degree-2 polynomial ridge regression from the level. Included as a constrained benchmark — slope is inferred rather than observed since only the 3M is available at test time.

---

## Results

### Calibrated Parameters

| Model | κ (kappa) | θ (theta) | σ (sigma) | Feller Margin (2κθ − σ²) |
|-------|-----------|-----------|-----------|--------------------------|
| OLS dynamics CIR | 0.000252 | 9.3589 | 0.036099 | 0.003414 |
| Cross-sectional CIR | 0.166010 | 0.024410 | 0.000028 | 0.008105 |

> **Feller condition satisfied** for the cross-sectional CIR calibration (Feller margin > 0).

The two methods produce very different parameter values because they optimise for different objectives: OLS fits the time-series dynamics of the short rate, while cross-sectional calibration fits the shape of the entire yield curve.

---

### Inter-Tenor Correlation

Pearson correlation matrix of training yields (all tenors):

| | ZC025YR | ZC050YR | ZC075YR | ZC100YR | ZC200YR | ZC500YR | ZC1000YR | ZC2000YR | ZC3000YR |
|---|---|---|---|---|---|---|---|---|---|
| **ZC025YR** | 1.000 | 0.997 | 0.992 | 0.984 | 0.959 | 0.903 | 0.856 | 0.813 | 0.805 |
| **ZC050YR** | 0.997 | 1.000 | 0.999 | 0.995 | 0.976 | 0.928 | 0.882 | 0.837 | 0.829 |
| **ZC075YR** | 0.992 | 0.999 | 1.000 | 0.999 | 0.985 | 0.943 | 0.897 | 0.851 | 0.842 |
| **ZC100YR** | 0.984 | 0.995 | 0.999 | 1.000 | 0.992 | 0.955 | 0.910 | 0.862 | 0.853 |
| **ZC200YR** | 0.959 | 0.976 | 0.985 | 0.992 | 1.000 | 0.983 | 0.943 | 0.890 | 0.881 |
| **ZC500YR** | 0.903 | 0.928 | 0.943 | 0.955 | 0.983 | 1.000 | 0.982 | 0.931 | 0.925 |
| **ZC1000YR** | 0.856 | 0.882 | 0.897 | 0.910 | 0.943 | 0.982 | 1.000 | 0.977 | 0.976 |
| **ZC2000YR** | 0.813 | 0.837 | 0.851 | 0.862 | 0.890 | 0.931 | 0.977 | 1.000 | 0.997 |
| **ZC3000YR** | 0.805 | 0.829 | 0.842 | 0.853 | 0.881 | 0.925 | 0.976 | 0.997 | 1.000 |

---

### Base CIR — Per-Tenor Test Metrics

Out-of-sample metrics on `test_data.csv`, predicting 6M/9M/1Y/2Y from the observed 3M rate:

| Tenor | Maturity (yrs) | R² | RMSE (bp) | MAE (bp) | Bias (bp) |
|-------|---------------|-----|-----------|----------|-----------|
| **OVERALL** | — | 0.8924 | 21.96 | 14.20 | 4.95 |
| ZC050YR (6M) | 0.5 | 0.9942 | 5.98 | 4.36 | 1.36 |
| ZC075YR (9M) | 0.75 | 0.9675 | 13.01 | 9.68 | 4.40 |
| ZC100YR (1Y) | 1.0 | 0.9101 | 19.73 | 14.72 | 6.33 |
| ZC200YR (2Y) | 2.0 | 0.3889 | 36.53 | 28.04 | 7.71 |

> The per-tenor breakdown identifies exactly where the one-factor CIR curve succeeds and where it struggles.

---

### Full Model Comparison

Out-of-sample comparison of all four models on `test_data.csv`:

| Model | Tenor | Maturity (yrs) | R² | RMSE (bp) | MAE (bp) | Bias (bp) |
|-------|-------|---------------|-----|-----------|----------|-----------|
| **Base CIR** | OVERALL | — | 0.8924 | 21.96 | 14.20 | 4.95 |
| | ZC050YR | 0.5 | 0.9942 | 5.98 | 4.36 | 1.36 |
| | ZC075YR | 0.75 | 0.9675 | 13.01 | 9.68 | 4.40 |
| | ZC100YR | 1.0 | 0.9101 | 19.73 | 14.72 | 6.33 |
| | ZC200YR | 2.0 | 0.3889 | 36.53 | 28.04 | 7.71 |
| **CIR++** | OVERALL | — | 0.6761 | 38.10 | 33.32 | 31.47 |
| | ZC050YR | 0.5 | 0.8888 | 26.26 | 25.05 | 25.05 |
| | ZC075YR | 0.75 | 0.6973 | 39.72 | 37.24 | 37.24 |
| | ZC100YR | 1.0 | 0.3719 | 52.16 | 48.32 | 48.32 |
| | ZC200YR | 2.0 | 0.6251 | 28.62 | 22.66 | 15.29 |
| **Level-Slope** | OVERALL | — | 0.6946 | 37.00 | 32.49 | 30.89 |
| | ZC050YR | 0.5 | 0.9107 | 23.53 | 22.53 | 22.53 |
| | ZC075YR | 0.75 | 0.7372 | 37.01 | 34.73 | 34.73 |
| | ZC100YR | 1.0 | 0.4350 | 49.47 | 45.81 | 45.81 |
| | ZC200YR | 2.0 | 0.4940 | 33.25 | 26.89 | 20.51 |
| **Selective CIR++**  | OVERALL | — | **0.9431** | **15.97** | **11.02** | 2.17 |
| | ZC050YR | 0.5 | 0.9942 | 5.98 | 4.36 | 1.36 |
| | ZC075YR | 0.75 | 0.9675 | 13.01 | 9.68 | 4.40 |
| | ZC100YR | 1.0 | 0.9101 | 19.73 | 14.72 | 6.33 |
| | ZC200YR | 2.0 | **0.8048** | **20.65** | **15.32** | −3.42 |

> NOTE: **Selected model: Selective rate-time CIR++** — best overall R² and RMSE on test data.

---

### 3M-Only Test Predictions (Sample)

Using only the 3M yield from `test_data_3M.csv`, the Selective CIR++ model reconstructs the full yield curve. First 5 rows of predicted output:

| Date | ZC025YR (input) | ZC050YR | ZC075YR | ZC100YR | ZC200YR | ZC500YR | ZC1000YR | ZC2000YR | ZC3000YR |
|------|----------------|---------|---------|---------|---------|---------|----------|----------|----------|
| 2024-04-29 | 0.049144 | 0.048146 | 0.047667 | 0.047200 | 0.040151 | 0.033875 | 0.034131 | 0.034163 | 0.033490 |
| 2024-04-30 | 0.049156 | 0.048157 | 0.047678 | 0.047212 | 0.040144 | 0.033870 | 0.034132 | 0.034166 | 0.033495 |
| 2024-05-01 | 0.049100 | 0.048104 | 0.047625 | 0.047160 | 0.040098 | 0.033839 | 0.034109 | 0.034148 | 0.033480 |
| 2024-05-02 | 0.048921 | 0.047931 | 0.047456 | 0.046994 | 0.039979 | 0.033759 | 0.034044 | 0.034092 | 0.033430 |
| 2024-05-03 | 0.048633 | 0.047655 | 0.047186 | 0.046729 | 0.039798 | 0.033638 | 0.033941 | 0.034002 | 0.033349 |

Predictions are saved to `outputs/predicted_yield_curve_from_test_3M.csv`.

---

## Key Inferences

The following inferences are drawn directly from the notebook's critical analysis:

1. **Calibration method sensitivity** — The calibrated yield curve is highly sensitive to the chosen methodology. OLS optimises for day-to-day short-rate persistence but ignores the broader term structure shape. Cross-sectional calibration minimises error across all tenors simultaneously and produces vastly superior full-curve reconstruction.

2. **Feller condition in practice** — The Feller condition ($2\kappa\theta \geq \sigma^2$) can break down in near-zero rate environments or during market stress when volatility spikes relative to the mean level. This is handled by: (a) flooring raw yields at $10^{-6}$ to prevent negative square-root inputs, and (b) imposing a structural penalty in the cross-sectional loss that discourages — but does not prohibit — Feller violations.

3. **Meaning of mean reversion speed** — $\kappa$ inversely correlates with shock persistence. A higher $\kappa$ implies unexpected shocks decay rapidly back to $\theta$, a lower $\kappa$ implies shocks are persistent and rates wander away from the mean for extended periods.

4. **Accuracy by maturity** — The 3M rate reconstructs the short end (6M, 9M, 1Y) with high accuracy since those tenors are fundamentally anchored to the 3M level. The **2Y tenor is the hardest to fit** (base CIR R² = 0.39) because it encodes slope and term-premium information that a single-factor model cannot observe from the 3M proxy alone.

5. **Systematic errors of the base model** — A one-factor CIR framework forces the entire yield curve to move in tandem with a single $r_t$. It cannot perfectly fit an arbitrary initial term structure and constrains the curve to shapes a single factor can produce. In steep curve environments, it systematically underestimates long yields; in inverted environments, it overestimates them.

6. **Value of the Selective CIR++ extension** — The Selective rate-time CIR++ raises overall R² from 0.8924 to 0.9431 and cuts overall RMSE from 21.96 bp to 15.97 bp without overfitting. It achieves this by leaving the already-accurate short-end predictions (6M, 9M, 1Y) entirely unchanged and applying a recency-weighted deterministic shift only to the problematic 2Y+ sector.

7. **Why CIR++ (global residual correction) underperforms** — A global polynomial residual fit on the full rate history can overfit older rate regimes. When term premiums shift, longer-term yields can move independently of the short rate, which the global correction cannot handle. This causes it to perform worse than the base CIR on the short end despite nominally correcting residuals.

8. **Jump processes** — Incorporating jump processes into the SDE would account for sudden policy announcements or macro shocks, introducing fatter tails and discontinuous steps in the short rate. A standard continuous CIR model would take days or weeks to diffuse to the same level that a jump-diffusion model prices in instantly.

9. **Two-factor model estimation challenges** — A two-factor model introduces a second stochastic factor to separately capture level and slope, but those factors are typically latent (unobservable), requiring computationally heavy state-space techniques such as Kalman Filtering. Time-dependent shift functions also risk temporal overfitting — fitting the calibration date perfectly while producing unstable forward rates.

---

## Dependencies

```python
numpy
pandas
scipy
matplotlib
```

Run in **Google Colab** or any standard Python 3 environment. Upload `train_data.csv`, `test_data.csv`, and `test_data_3M.csv` before executing the notebook.

Refer: https://colab.research.google.com/drive/1gRbwlbpq8C6Hmr1KbJVqsI8fYi46NoUP?usp=sharing
