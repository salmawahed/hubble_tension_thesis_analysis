"""
run_analysis.py
---------------
top-level entry-point script. runs the full analysis pipeline in sequence:

    load data -> MLE -> MCMC -> diagnostics -> posterior summary
    -> tension metrics -> all figures

running this single script reproduces every numerical result and figure
in Sections 4.4-4.6 of the thesis.

usage:
    python run_analysis.py

expected runtime: 30-90 minutes (dominated by MCMC sampling).

Thesis: The Hubble Tension: Mathematical Foundations,
        Statistical Analysis, and Proposed Resolutions
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.linalg import cho_factor

from model import (comoving_distance, build_chi_cache, interpolate_chi,
                   luminosity_distance, distance_modulus)
from likelihood import build_covariance, log_posterior
from MLE import run_mle, print_mle_summary
from MCMC import (run_sampler, get_flat_samples, autocorr_time,
                  gelman_rubin, posterior_summary, tension_metric,
                  p_planck, print_mcmc_summary, save_chain)
from plots import (plot_1d_posterior, plot_2d_posterior, plot_trace,
                   plot_autocorr, plot_corner, plot_h0_comparison)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_FILE = "Pantheon+SH0ES_data.dat"
COV_FILE  = "Pantheon+SH0ES_STAT+SYS.cov"

H0_GRID   = np.linspace(68.0, 78.0, 200)
OM_GRID   = np.linspace(0.20, 0.45, 50)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("\n[DATA] Loading Pantheon+SH0ES dataset...")
data = pd.read_csv(DATA_FILE, sep=r'\s+')
data = data.replace([np.inf, -np.inf], float('nan'))
data = data.dropna(subset=["zHD", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"])
mask = data["zHD"].values > 1e-4
data = data[mask].reset_index(drop=True)
N      = len(data)
z      = data["zHD"].values
mu_obs = data["MU_SH0ES"].values
mu_err = data["MU_SH0ES_ERR_DIAG"].values
print(f"[DATA] Using {N} supernovae (full Pantheon+ sample)")

# ── 2. Load Covariance Matrix ─────────────────────────────────────────────────
print("[COV]  Loading covariance matrix...")
with open(COV_FILE, "r") as f:
    n_cov   = int(f.readline().strip())
    C_full  = np.array(f.read().split(), dtype=float).reshape(n_cov, n_cov)

original = pd.read_csv(DATA_FILE, sep=r'\s+')
original = original.replace([np.inf, -np.inf], float('nan'))
original = original.dropna(subset=["zHD", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"])
idx   = np.where(original["zHD"].values > 1e-4)[0]
C_sys = C_full[np.ix_(idx, idx)]

_, cho, _, const = build_covariance(C_sys, mu_err)
print(f"[COV]  Covariance matrix built: {N} x {N}")

# ── 3. Pre-Compute Distance Integrals ─────────────────────────────────────────
print("[MODEL] Pre-computing comoving integrals...")
Om_cache_arr  = np.linspace(0.10, 0.70, 300)
chi_cache_arr = np.zeros((len(Om_cache_arr), N))
for k, Om in enumerate(Om_cache_arr):
    chi_cache_arr[k] = comoving_distance(z, Om)
    if k % 50 == 0:
        print(f"  Om={Om:.3f}  ({k}/{len(Om_cache_arr)})", flush=True)
print("[MODEL] Integrals ready.")

# ── 4. Frequentist MLE ────────────────────────────────────────────────────────
print("\n[MLE]  Running L-BFGS-B optimisation...")
result, H0_mle, Om_mle = run_mle(
    z, mu_obs, cho, const, Om_cache_arr, chi_cache_arr
)
print_mle_summary(H0_mle, Om_mle)

# ── 5. Bayesian Grid (1D Marginal) ────────────────────────────────────────────
print("\n[GRID] Evaluating 2D likelihood grid for 1D marginal posterior...")
log_like_2D = np.zeros((len(H0_GRID), len(OM_GRID)))
for i, H0 in enumerate(H0_GRID):
    if i % 50 == 0:
        print(f"  H0={H0:.1f}  ({i}/{len(H0_GRID)})", flush=True)
    for j, Om in enumerate(OM_GRID):
        chi   = interpolate_chi(Om, Om_cache_arr, chi_cache_arr)
        dl    = luminosity_distance(z, H0, chi)
        mu_th = distance_modulus(dl)
        resid = mu_obs - mu_th
        from scipy.linalg import cho_solve
        Cinv_r = cho_solve(cho, resid)
        log_like_2D[i, j] = const - 0.5 * resid @ Cinv_r

log_like_2D -= log_like_2D.max()
like_2D      = np.exp(log_like_2D)

dOm         = OM_GRID[1] - OM_GRID[0]
marginal_H0 = np.sum(like_2D, axis=1) * dOm
dH0         = H0_GRID[1] - H0_GRID[0]
marginal_H0 /= np.sum(marginal_H0) * dH0

cdf  = np.cumsum(marginal_H0) * dH0
cdf /= cdf[-1]

H0_MAP_grid  = H0_GRID[np.argmax(marginal_H0)]
h0_low_68    = H0_GRID[np.searchsorted(cdf, 0.16)]
h0_high_68   = H0_GRID[np.searchsorted(cdf, 0.84)]
h0_low_95    = H0_GRID[np.searchsorted(cdf, 0.025)]
h0_high_95   = H0_GRID[np.searchsorted(cdf, 0.975)]
sigma_lo_grid = H0_MAP_grid - h0_low_68
sigma_hi_grid = h0_high_68  - H0_MAP_grid

print(f"[GRID] H0 MAP = {H0_MAP_grid:.2f} "
      f"+{sigma_hi_grid:.2f} / -{sigma_lo_grid:.2f} km/s/Mpc")

# ── 6. MCMC ───────────────────────────────────────────────────────────────────
print("\n[MCMC] Running emcee sampler (this takes 30-90 minutes)...")
sampler = run_sampler(z, mu_obs, cho, const, Om_cache_arr, chi_cache_arr)
save_chain(sampler)

chain        = sampler.get_chain()
flat_samples = get_flat_samples(sampler)
tau          = autocorr_time(sampler)
rhat         = gelman_rubin(chain)
summary      = posterior_summary(flat_samples)
n_eff        = flat_samples.shape[0]
H0_mcmc      = summary['H0']['median']
tension      = tension_metric(H0_mcmc)
p_val        = p_planck(flat_samples[:, 0])

print_mcmc_summary(summary, tau, rhat, n_eff, tension)
print(f"\n[TENS] P(H0 <= 67.4 | data) = {p_val:.6f}")

# ── 7. 2D Posterior for Corner/Joint Plot ─────────────────────────────────────
print("\n[2D]  Computing 2D posterior for joint plot...")
H0_GRID_2D = np.linspace(70.5, 75.5, 150)
OM_GRID_2D = np.linspace(0.10, 0.70, 150)
log_like_2D_fine = np.zeros((len(H0_GRID_2D), len(OM_GRID_2D)))
for i, H0 in enumerate(H0_GRID_2D):
    for j, Om in enumerate(OM_GRID_2D):
        chi   = interpolate_chi(Om, Om_cache_arr, chi_cache_arr)
        dl    = luminosity_distance(z, H0, chi)
        mu_th = distance_modulus(dl)
        resid = mu_obs - mu_th
        from scipy.linalg import cho_solve
        Cinv_r = cho_solve(cho, resid)
        log_like_2D_fine[i, j] = const - 0.5 * resid @ Cinv_r

log_like_2D_fine -= log_like_2D_fine.max()
post_2D = np.exp(log_like_2D_fine)
dH0_2D  = H0_GRID_2D[1] - H0_GRID_2D[0]
dOm_2D  = OM_GRID_2D[1] - OM_GRID_2D[0]
post_2D /= post_2D.sum() * dH0_2D * dOm_2D

post_plot = post_2D / post_2D.max()
flat_s = np.sort(post_plot.ravel())[::-1]
cum    = np.cumsum(flat_s) * dH0_2D * dOm_2D
cum   /= cum[-1]
level_68 = flat_s[np.searchsorted(cum, 0.68)]
level_95 = flat_s[np.searchsorted(cum, 0.95)]
map_idx  = np.unravel_index(post_2D.argmax(), post_2D.shape)
H0_MAP_2D = H0_GRID_2D[map_idx[0]]
Om_MAP_2D = OM_GRID_2D[map_idx[1]]

# ── 8. Generate All Figures ───────────────────────────────────────────────────
print("\n[FIGS] Generating all figures...")

plot_1d_posterior(
    H0_GRID, marginal_H0, H0_MAP_grid,
    h0_low_68, h0_high_68, h0_low_95, h0_high_95,
    sigma_lo_grid, sigma_hi_grid
)

plot_2d_posterior(
    H0_GRID_2D, OM_GRID_2D, post_2D,
    level_68, level_95, H0_MAP_2D, Om_MAP_2D
)

plot_trace(chain)
plot_autocorr(chain)
plot_corner(flat_samples)

plot_h0_comparison(
    H0_this_work_mcmc=summary['H0']['median'],
    H0_this_work_mle=H0_mle,
    sigma_mcmc_lo=summary['H0']['lo'],
    sigma_mcmc_hi=summary['H0']['hi'],
    sigma_mle=0.37
)

# ── 9. Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL RESULTS")
print("=" * 55)
print(f"  H0  (MLE)  : {H0_mle:.2f} +/- 0.37 km/s/Mpc")
print(f"  H0  (MCMC) : {summary['H0']['median']:.2f} "
      f"+{summary['H0']['hi']:.2f} / -{summary['H0']['lo']:.2f} km/s/Mpc")
print(f"  Om  (MLE)  : {Om_mle:.3f}")
print(f"  Om  (MCMC) : {summary['Om']['median']:.3f} "
      f"+{summary['Om']['hi']:.3f} / -{summary['Om']['lo']:.3f}")
print(f"  tau (H0)   : {tau[0]:.1f} steps")
print(f"  tau (Om)   : {tau[1]:.1f} steps")
print(f"  N_eff      : ~{n_eff}")
print(f"  R-hat (H0) : {rhat[0]:.4f}")
print(f"  R-hat (Om) : {rhat[1]:.4f}")
print(f"  Tension    : ~{tension:.1f} sigma")
print(f"  P(H0<=67.4): {p_val:.6f}")
print("=" * 55)
print("\nAll figures saved to figures/")
print("Done.")