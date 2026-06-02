# Hubble Tension

**Graduation Thesis: *The Hubble Tension: Mathematical Foundations, Statistical Analysis, and Proposed Resolutions***

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

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
```
## Quickstart##

 1. Clone the repository```bashgit clone https://github.com/salmawahed/hubble-tension.gitcd hubble-tension```> **Thesis version:** To reproduce the exact results in the thesis, checkout the> tagged release:> ```bash> git checkout v1.0-thesis-submission> ```### 2. Set up the environment**Using conda (recommended):**```bashconda env create -f environment.ymlconda activate hubble-tension-mcmc```**Using pip:**```bashpip install -r requirements.txt```### 3. Download the dataThe Pantheon+SH0ES dataset is publicly available from the SH0ES team.See [`data/README.md`](data/README.md) for download instructions.Place the following files in the `data/` directory:```data/├── Pantheon+SH0ES.dat└── Pantheon+SH0ES_STAT+SYS.cov```### 4. Run the full analysis```bashpython run_analysis.py data/Pantheon+SH0ES.dat```**Expected runtime:** 30–90 minutes (dominated by the MCMC sampling).**Expected output:**```[DATA] Total SNe in file: 1701[DATA] Calibrator SNe excluded: 277[DATA] After redshift cuts: [N][MLE] H0 = 73.0100 km/s/Mpc[MLE] Omega_m = 0.3340[MCMC] Starting 32 walkers × 5000 steps...[POST] H0 = 73.04 +0.37 -0.36 km/s/Mpc[POST] Om = 0.346 +0.029 -0.027[TENS] Gaussian tension T: 4.9 sigma```All figures are saved to `figures/`.---## Output Figures| File | Description | Thesis figure ||------|-------------|---------------|| `fig3_1_hubble_diagram.pdf` | Hubble diagram + residuals | Figure 3.1 || `fig3_2_posterior_1d.pdf` | 1D marginal posterior P(H₀\|D) | Figure 3.2 || `fig3_3_joint_posterior.pdf` | 2D joint posterior P(H₀, Ω_m\|D) | Figure 3.3 || `fig3_4_trace.pdf` | MCMC trace plots (all walkers) | Figure 3.4 || `fig3_5_autocorr.pdf` | Autocorrelation function | Figure 3.5 || `fig3_6_corner.pdf` | Corner plot | Figure 3.6 || `fig3_7_comparison.pdf` | H₀ comparison whisker plot | Figure 3.7 |---## Physics BackgroundThe theoretical model is the **flat ΛCDM luminosity distance**:```D_L(z; H₀, Ω_m) = (1+z) × (c/H₀) × ∫₀ᶻ dz' / E(z')E(z) = √[ Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_Λ ]```with `Ω_Λ = 1 − Ω_m − Ω_r` (flat geometry) and `Ω_r ≈ 9.4 × 10⁻⁵` fixed.The **distance modulus** is:```μ_th(z; H₀, Ω_m) = 5 log₁₀[D_L / Mpc] + 25```This is the formula derived from first principles in Section 2.6.3 of the thesis(starting from the FLRW metric and the Friedmann equations).### Selection cuts applied to the data| Cut | Criterion | Motivation ||-----|-----------|------------|| Remove calibrators | `IS_CALIBRATOR = 0` | Avoid circularity with M_B anchor || Lower z cut | `zHD ≥ 0.023` | Suppress peculiar velocity contamination || Upper z cut | `zHD ≤ 0.15` | Stay in linear Hubble flow regime |### MCMC configuration| Parameter | Value | Reason ||-----------|-------|--------|| Walkers | 32 | Standard for 2D parameter space || Steps | 5000 | >> 50τ, ensures convergence || Burn-in | 500 | Conservative; chain converges in ~100 || Thinning | 15 | Reduces autocorrelation; N_eff ≈ 9600 || Seed | 42 | Reproducibility || Priors | H₀ ~ U(40,100), Ω_m ~ U(0.05,0.7) | Uninformative |---## Key Results```━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ FINAL RESULTS━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ H₀ (MLE): 73.01 ± 0.37 km/s/Mpc H₀ (MCMC): 73.04 +0.37/−0.36 km/s/Mpc Ω_m (MLE): 0.334 Ω_m (MCMC): 0.346 +0.029/−0.027 τ (H₀): 32.7 steps τ (Ω_m): 32.6 steps N_eff: ≈ 9600 N_steps/τ: ≈ 153 (convergence criterion: >> 50 ✓)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Planck 2018: 67.4 ± 0.5 km/s/Mpc Tension: ~4.9σ (p ≈ 4.8 × 10⁻⁷)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━```---## Dependencies| Package | Version | Purpose ||---------|---------|---------|| Python | 3.11.x | Runtime || numpy | 1.26.x | Arrays and statistics || scipy | 1.11.x | Integration (quad) and optimization (minimize) || emcee | 3.1.4 | Affine-invariant MCMC sampler || corner | 2.2.2 | Corner plot visualization || matplotlib | 3.8.x | All figures || jupyter | 1.0.0 | Interactive notebooks || tqdm | 4.66.x | MCMC progress bar |---## DataThe **Pantheon+SH0ES** dataset is the work of:- Scolnic, D. et al. (2022). *The Pantheon+ Analysis: The Full Data Set and Light-Curve Release.* ApJ 938, 113. [arXiv:2112.03863](https://arxiv.org/abs/2112.03863)- Brout, D. et al. (2022). *The Pantheon+ Analysis: Cosmological Constraints.* ApJ 938, 110. [arXiv:2202.04077](https://arxiv.org/abs/2202.04077)- Riess, A. G. et al. (2022). *A Comprehensive Measurement of the Local Value of the Hubble Constant.* ApJL 934, L7. [arXiv:2112.04510](https://arxiv.org/abs/2112.04510)Data download: [https://github.com/PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease)--- for details.---## Contact Salma Wahed · Alexanderia University · salmawahedscholar@gmail.com *Thesis submitted [June, 2026] in partial fulfillment of the requirements for the bachelor of science in Physics.

