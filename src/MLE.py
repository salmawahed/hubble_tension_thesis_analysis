"""
mle.py
------
frequentist maximum likelihood estimation for H0 and Omega_m.

uses scipy.optimize.minimize with the L-BFGS-B method to minimize
the negative log-likelihood (equivalent to chi-squared minimization
when the covariance matrix is fixed).

computes:
  - MLE point estimates for H0 and Omega_m
  - reduced chi-squared statistic
  - profile likelihood confidence intervals on H0

Thesis: The Hubble Tension: Mathematical Foundations,
        Statistical Analysis, and Proposed Resolutions
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_solve
from model import (comoving_distance, luminosity_distance,
                   distance_modulus, interpolate_chi)


def neg_log_likelihood(theta, z, mu_obs, cho, const,
                       Om_cache, chi_cache):
    """
    negative log-likelihood for minimisation.

    parameters
    ----------
    (same as likelihood.log_likelihood)

    returns
    -------
    float — negative log-likelihood
    """
    H0, Om = theta
    # reject out-of-bounds parameters
    if H0 <= 0 or Om <= 0 or Om >= 1:
        return np.inf
    chi   = interpolate_chi(Om, Om_cache, chi_cache)
    dl    = luminosity_distance(z, H0, chi)
    mu_th = distance_modulus(dl)
    resid = mu_obs - mu_th
    Cinv_r = cho_solve(cho, resid)
    return -(const - 0.5 * resid @ Cinv_r)


def run_mle(z, mu_obs, cho, const, Om_cache, chi_cache,
            H0_init=73.0, Om_init=0.35):
    """
    run L-BFGS-B optimisation to find the MLE estimates.

    parameters
    ----------
    z, mu_obs        : data arrays
    cho, const       : cholesky factorization and normalization constant
    Om_cache, chi_cache : cached integrals for interpolation
    H0_init, Om_init : initial parameter guesses

    returns
    -------
    result : scipy OptimizeResult object
    H0_mle : float — MLE estimate of H0
    Om_mle : float — MLE estimate of Omega_m
    """
    result = minimize(
        neg_log_likelihood,
        x0=[H0_init, Om_init],
        args=(z, mu_obs, cho, const, Om_cache, chi_cache),
        method='L-BFGS-B',
        bounds=[(60.0, 85.0), (0.10, 0.70)],
        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 10000}
    )

    H0_mle, Om_mle = result.x
    return result, H0_mle, Om_mle


def reduced_chi_squared(theta, z, mu_obs, cho, N_params=2):
    """
    compute the reduced chi-squared statistic at the MLE.

    chi2_red = chi2 / (N_data - N_params)

    parameters
    ----------
    theta    : array-like — [H0, Om] at MLE
    z        : redshifts
    mu_obs   : observed distance moduli
    cho      : Cholesky factorization of C
    N_params : int — number of free parameters (default 2)

    returns
    -------
    chi2     : float — total chi-squared
    chi2_red : float — reduced chi-squared
    """
    from model import interpolate_chi
    H0, Om = theta
    chi   = interpolate_chi(Om, *cho)   # placeholder — pass Om_cache, chi_cache
    dl    = luminosity_distance(z, H0, chi)
    mu_th = distance_modulus(dl)
    resid = mu_obs - mu_th
    Cinv_r = cho_solve(cho, resid)
    chi2 = resid @ Cinv_r
    N = len(z)
    return chi2, chi2 / (N - N_params)


def profile_likelihood_H0(H0_grid, z, mu_obs, cho, const,
                           Om_cache, chi_cache):
    """
    compute the profile likelihood for H0 by maximizing
    over Omega_m at each fixed H0.

    this gives the marginalized frequentist confidence interval
    on H0 without assuming Omega_m is known.

    parameters
    ----------
    H0_grid : array-like — H0 values to evaluate
    (others) : same as run_mle()

    returns
    -------
    profile : numpy.ndarray — profile log-likelihood values
    Om_best : numpy.ndarray — best-fit Omega_m at each H0
    """
    profile = np.empty(len(H0_grid))
    Om_best = np.empty(len(H0_grid))

    for i, H0 in enumerate(H0_grid):
        res = minimize(
            lambda Om: neg_log_likelihood(
                [H0, Om[0]], z, mu_obs, cho, const,
                Om_cache, chi_cache
            ),
            x0=[0.35],
            method='L-BFGS-B',
            bounds=[(0.10, 0.70)],
        )
        profile[i] = -res.fun
        Om_best[i] = res.x[0]

    # normalise so peak = 0
    profile -= profile.max()
    return profile, Om_best


def print_mle_summary(H0_mle, Om_mle, H0_err=None, tension=None):
    """print a formatted MLE results summary."""
    print("\n" + "=" * 50)
    print("  FREQUENTIST MLE RESULTS")
    print("=" * 50)
    print(f"  H0      = {H0_mle:.4f} km/s/Mpc")
    if H0_err:
        print(f"  H0 err  = +/- {H0_err:.4f} km/s/Mpc")
    print(f"  Omega_m = {Om_mle:.4f}")
    if tension:
        print(f"  Tension = {tension:.2f} sigma")
    print("=" * 50)

