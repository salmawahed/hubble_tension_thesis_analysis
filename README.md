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
│   ├── 02_frequentist_mle.ipynb
│   ├── 03_bayesian_mcmc.ipynb
│   └── 04_tension_analysis.ipynb
│
└── run_analysis.py                # single entry-point: runs everything
```


