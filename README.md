# Hubble Tension

**Graduation Thesis: *The Hubble Tension: Mathematical Foundations, Statistical Analysis, and Proposed Resolutions***

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Thesis Release](https://img.shields.io/badge/release-v1.0--thesis--submission-orange.svg)](https://github.com/salmawahed/hubble-tension-thesis-analysis/releases/tag/v1.0-thesis-submission)
 
---

## Overview
This repository contains the complete statistical analysis pipeline for the Hubble constant measurement presented in Chapter 4 of the thesis. Using the publicly available **Pantheon+SH0ES** Type Ia supernova dataset (Scolnic et al. 2022, Riess et al. 2022), the code applies two independent statistical frameworks to the Hubble diagram to extract a value of $H_0$ and quantify the tension with the Planck CMB measurement.

| Method | $H_0$ result | vs Planck 2018 |
|--------|-----------|----------------|
| Frequentist MLE | **73.01 ± 0.37 km/s/Mpc** | ~4.9σ tension |
| Bayesian MCMC (median) | **73.04 +0.37/−0.36 km/s/Mpc** | ~4.9σ tension |
| Planck 2018 (reference) | 67.4 ± 0.5 km/s/Mpc | — |
| SH0ES 2022 (reference) | 73.04 ± 1.04 km/s/Mpc | — |

The analysis reproduces the published SH0ES 2022 result to within 0.04%, confirming the implementation is correct.

---

## Repository Structure
```hubble-tension-mcmc/
│
├── README.md                      # this file
├── requirements.txt               # pinned pip dependencies
├── environment.yml                # conda environment
│
├── src/
│   ├── model.py                   # flat ΛCDM luminosity distance
│   ├── likelihood.py              # log-likelihood, log-prior, log-posterior
│   ├── mle.py                     # frequentist chi-squared minimisation
│   ├── mcmc.py                    # emcee sampler + convergence diagnostics
│   └── plots.py                   # all figure generation functions
│
├── data/
│   └── README.md                  # instructions for downloading Pantheon+SH0ES
│
├── figures/                       # auto-populated when you run the analysis
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_frequentist_mle.ipynb
│   ├── 03_bayesian_mcmc.ipynb
│   └── 04_tension_analysis.ipynb
│
└── run_analysis.py                # single entry-point: runs everything
```
## Quickstart

### 1. Clone the repository
```bash
git clone [https://github.com/salmawahed/hubble_tension_thesis_analysis]
cd hubble-tension-thesis-analysis
```
### 2. Set up the environment
 
```bash
pip install -r requirements.txt
```
 
### 3. Download the data
 
The Pantheon+SH0ES dataset is publicly available from the SH0ES team.
Place the following files in the project directory:
 
```
Pantheon+SH0ES_data.dat
Pantheon+SH0ES_STAT+SYS.cov
```
 
Download from: [https://github.com/PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease)

 ### 4. Run the analysis scripts in order
 
```bash
# Step 1 — 1D marginal posterior for H₀ (grid method)
python H0_posterior_cov.py
 
# Step 2 — 2D joint posterior + Ωm marginalisation
python H0_Om_2D_posterior.py
 
# Step 3 — MCMC sampling (saves mcmc_chain.npy)
python mcmc_trace.py
 
# Step 4 — Autocorrelation function (requires mcmc_chain.npy)
python mcmc_autocorr.py
 
# Step 5 — Corner plot (requires mcmc_chain.npy)
python mcmc_corner.py
 
# Step 6 — H₀ comparison whisker plot
python h0_comparison.py
```
 
> **Note:** Step 3 (MCMC) takes 15–90 minutes depending on your machine.
> All other steps run in under 5 minutes.

---
 
## Key Results
 
```
══════════════════════════════════════════════════════
  FINAL RESULTS
══════════════════════════════════════════════════════
  H₀  (MLE)  :  73.01 ± 0.37         km/s/Mpc
  H₀  (MCMC) :  73.04 +0.37/−0.36   km/s/Mpc
  Ω_m (MLE)  :  0.334
  Ω_m (MCMC) :  0.346 +0.029/−0.027
  τ (H₀)     :  32.7 steps
  τ (Ω_m)    :  32.6 steps
  N_eff       :  ≈ 9600
  R̂ (H₀)    :  1.0038   (convergence threshold < 1.01 ✓)
  R̂ (Ω_m)   :  1.0036   (convergence threshold < 1.01 ✓)
  N_steps/τ  :  ≈ 153    (convergence criterion >> 50 ✓)
══════════════════════════════════════════════════════
  Planck 2018:   67.4 ± 0.5  km/s/Mpc
  Tension    :   ~4.9σ   (using full SH0ES error budget)
══════════════════════════════════════════════════════
```
 ---
 
## Output Figures
 
| File | Description |
|------|-------------|
| `H0_1D_Posterior_fullcov.png` | 1D marginal posterior P(H₀\|D) with Planck tension arrow |
| `H0_Om_2D_Posterior.png` | 2D joint posterior with 68% and 95% credible contours |
| `MCMC_trace.png` | Trace plots for all 32 walkers, burn-in marked |
| `MCMC_autocorr.png` | Autocorrelation function with 50τ verification |
| `MCMC_corner.png` | Corner plot with 1D marginals and 2D joint posterior |
| `H0_comparison.png` | H₀ comparison whisker plot (Early vs Late universe) |
 
---
 
## Physics Background
 
The theoretical model is the **flat ΛCDM luminosity distance**:
 
```
D_L(z; H₀, Ω_m) = (1+z) × (c/H₀) × ∫₀ᶻ dz' / E(z')
 
E(z) = √[ Ω_m(1+z)³ + Ω_Λ ]    with Ω_Λ = 1 − Ω_m
```
 
The **distance modulus** is:
 
```
μ_th(z; H₀, Ω_m) = 5 log₁₀[D_L / Mpc] + 25
```
 
### Likelihood
 
The full Pantheon+ log-likelihood with systematic covariance:
 
```
ln L(H₀, Ω_m) = −½ rᵀ C⁻¹ r
 
C = C_sys + diag(σ_stat²)
```
 
where `r = μ_obs − μ_th` and `C_sys` is the 1701×1701 systematic
covariance matrix provided by the Pantheon+ collaboration.
 
### MCMC Configuration
 
| Parameter | Value | Reason |
|-----------|-------|--------|
| Walkers | 32 | Standard for 2D parameter space |
| Steps | 5000 | >> 50τ — ensures convergence |
| Burn-in | 500 | Conservative; chains converge in ~200 steps |
| Thinning | 15 | Reduces autocorrelation; N_eff ≈ 9600 |
| Seed | 42 | Reproducibility |
| Prior on H₀ | U(60, 85) | Flat, physically motivated |
| Prior on Ω_m | U(0.10, 0.70) | Flat, physically motivated |
 
### Tension Calculation
 
The Gaussian tension metric used throughout:
 
```
T = |H₀(this work) − H₀(Planck)| / √(σ_SH0ES² + σ_Planck²)
  = |73.04 − 67.4| / √(1.04² + 0.5²)
  ≈ 4.9σ
```
 
The full SH0ES uncertainty σ = 1.04 km/s/Mpc is used in the denominator
(rather than the tighter SNe-only uncertainty) as it represents the complete
distance ladder error budget and is the standard convention in the literature.
 
---
 
## Dependencies
 
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.x | Arrays and statistics |
| scipy | 1.11.x | Integration (quad) and Cholesky decomposition |
| emcee | 3.1.4 | Affine-invariant MCMC sampler |
| corner | 2.2.2 | Corner plot visualisation |
| matplotlib | 3.8.x | All figures |
| tqdm | 4.66.x | MCMC progress bar |
| pandas | 2.1.x | Data loading and cleaning |
 
Install all dependencies:
 
```bash
pip install numpy scipy emcee corner matplotlib tqdm pandas
```
 
---
 
## Data
 
The **Pantheon+SH0ES** dataset is the work of:
 
- Scolnic, D. et al. (2022). *The Pantheon+ Analysis: The Full Data Set and Light-Curve Release.* ApJ 938, 113. [arXiv:2112.03863](https://arxiv.org/abs/2112.03863)
- Brout, D. et al. (2022). *The Pantheon+ Analysis: Cosmological Constraints.* ApJ 938, 110. [arXiv:2202.04077](https://arxiv.org/abs/2202.04077)
- Riess, A. G. et al. (2022). *A Comprehensive Measurement of the Local Value of the Hubble Constant.* ApJL 934, L7. [arXiv:2112.04510](https://arxiv.org/abs/2112.04510)
---
 
## Common Issues
 
**UnicodeEncodeError on Windows:**
All print statements use ASCII-safe characters. If you still encounter encoding
errors, add this at the top of any script:
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```
 
**MCMC takes too long:**
Reduce `N_steps = 500` for a quick test run. The convergence criterion
requires N_steps >> 50τ ≈ 1635, so 5000 is the recommended minimum.
 
**mcmc_chain.npy not found:**
Make sure `mcmc_trace.py` completed successfully and saved the chain.
The last two lines of `mcmc_trace.py` should be:
```python
np.save("mcmc_chain.npy", sampler.get_chain())
np.save("mcmc_log_prob.npy", sampler.get_log_prob())
```
 
---
 


