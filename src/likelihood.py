"""
likelihood.py
-------------
gaussian log-likelihood, flat uninformative log-prior,
and log-posterior for the flat ΛCDM Hubble diagram fit.

the log-posterior is the target distribution sampled by the MCMC.
keeping the likelihood separate from the model allows the same
physical model to be used with different statistical frameworks
without duplication.

Thesis: The Hubble Tension: Mathematical Foundations,
        Statistical Analysis, and Proposed Resolutions
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from model import luminosity_distance, distance_modulus, interpolate_chi

# ── PRIOR BOUNDS ──────────────────────────────────────────────────────────────
H0_MIN, H0_MAX = 60.0, 85.0     # km/s/Mpc — flat prior range
OM_MIN, OM_MAX = 0.10, 0.70     # dimensionless — flat prior range


def build_covariance(C_sys, mu_err):
    """
    build the total Pantheon+ covariance matrix.

    C_total = C_sys + diag(sigma_stat^2)

    the .cov file contains only the systematic covariance.
    the statistical errors must be added explicitly on the diagonal.

    parameters
    ----------
    C_sys  : numpy.ndarray — systematic covariance matrix (N x N)
    mu_err : numpy.ndarray — diagonal statistical errors (N,)

    returns
    -------
    C      : numpy.ndarray — total covariance matrix (N x N)
    cho    : tuple         — cholesky factorisation for fast solving
    log_det: float         — log|C| = 2 * sum(log diag(L))
    const  : float         — normalization constant of log-likelihood
    """
    N = len(mu_err)
    C = C_sys + np.diag(mu_err**2)
    cho = cho_factor(C, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(cho[0])))
    const = -0.5 * (N * np.log(2.0 * np.pi) + log_det)
    return C, cho, log_det, const


def log_likelihood(theta, z, mu_obs, cho, const,
                   Om_cache, chi_cache):
    """
    gaussian log-likelihood with full covariance matrix.

    log L = const - 0.5 * r^T C^{-1} r

    where r = mu_obs - mu_theory, solved via Cholesky decomposition.

    parameters
    ----------
    theta      : array-like    — [H0, Om]
    z          : numpy.ndarray — redshifts
    mu_obs     : numpy.ndarray — observed distance moduli
    cho        : tuple         — Cholesky factorisation of C
    const      : float         — log-likelihood normalisation constant
    Om_cache   : numpy.ndarray — cached Omega_m grid for interpolation
    chi_cache  : numpy.ndarray — cached comoving integrals

    returns
    -------
    float — log-likelihood value
    """
    H0, Om = theta
    chi   = interpolate_chi(Om, Om_cache, chi_cache)
    dl    = luminosity_distance(z, H0, chi)
    mu_th = distance_modulus(dl)
    resid = mu_obs - mu_th
    Cinv_r = cho_solve(cho, resid)
    return const - 0.5 * resid @ Cinv_r


def log_prior(theta):
    """
    flat (uninformative) log-prior on [H0, Om].

    returns 0.0 if parameters are within bounds, -inf otherwise.

    parameters
    ----------
    theta : array-like — [H0, Om]

    returns
    -------
    float — 0.0 or -inf
    """
    H0, Om = theta
    if H0_MIN < H0 < H0_MAX and OM_MIN < Om < OM_MAX:
        return 0.0
    return -np.inf


def log_posterior(theta, z, mu_obs, cho, const,
                  Om_cache, chi_cache):
    """
    log-posterior = log-prior + log-likelihood.

    this is the target distribution sampled by the MCMC.

    parameters
    ----------
    (same as log_likelihood, plus prior bounds applied first)

    returns
    -------
    float — log-posterior value
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(
        theta, z, mu_obs, cho, const, Om_cache, chi_cache
    )
