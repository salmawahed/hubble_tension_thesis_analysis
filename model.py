"""
model.py
--------
flat ΛCDM theoretical model for the Hubble diagram.

contains:
  - comoving_distance()   : numerical integration of 1/E(z)
  - luminosity_distance() : D_L = (1+z) * chi
  - distance_modulus()    : mu = 5*log10(D_L) + 25

this module is pure physics, no statistical machinery.
it is imported by likelihood.py, mle.py, and mcmc.py.

Thesis: The Hubble Tension: Mathematical Foundations,
        Statistical Analysis, and Proposed Resolutions
"""

import numpy as np
from scipy.integrate import quad

# speed of light in km/s
C_LIGHT = 299792.458


def E(z, Om):
    """
    dimensionless Hubble parameter for flat ΛCDM.

    E(z) = sqrt[ Omega_m * (1+z)^3 + Omega_Lambda ]

    with Omega_Lambda = 1 - Omega_m (flat geometry).

    parameters
    ----------
    z  : float — redshift
    Om : float — matter density parameter Omega_m

    returns
    -------
    float
    """
    OL = 1.0 - Om
    return np.sqrt(Om * (1.0 + z)**3 + OL)


def comoving_distance(z_arr, Om):
    """
    comoving distance chi(z) for an array of redshifts.

    chi(z) = (c/H0) * integral_0^z dz' / E(z')

    note: returns the dimensionless integral (c/H0 factor
    is applied later in luminosity_distance so H0 can vary).

    parameters
    ----------
    z_arr : array-like — redshifts
    Om    : float      — Omega_m

    returns
    -------
    numpy.ndarray — comoving distance integrals (c/H0 = 1 units)
    """
    z_arr = np.atleast_1d(z_arr)
    chi = np.empty(len(z_arr))
    for i, zi in enumerate(z_arr):
        chi[i], _ = quad(
            lambda zp: 1.0 / E(zp, Om),
            0.0, zi,
            limit=100
        )
    return chi


def luminosity_distance(z_arr, H0, chi):
    """
    luminosity distance D_L in Mpc.

    D_L(z; H0, Om) = (c/H0) * (1+z) * chi(z)

    parameters
    ----------
    z_arr : array-like     — redshifts
    H0    : float          — Hubble constant [km/s/Mpc]
    chi   : numpy.ndarray  — comoving integrals from comoving_distance()

    returns
    -------
    numpy.ndarray — luminosity distances [Mpc]
    """
    return (C_LIGHT / H0) * (1.0 + np.atleast_1d(z_arr)) * chi


def distance_modulus(dl):
    """
    theoretical distance modulus.

    mu_th = 5 * log10(D_L / Mpc) + 25

    parameters
    ----------
    dl : numpy.ndarray — luminosity distances [Mpc]

    returns
    -------
    numpy.ndarray — distance moduli [mag]
    """
    return 5.0 * np.log10(dl) + 25.0


def build_chi_cache(z_arr, Om_grid):
    """
    pre-compute comoving integrals for a grid of Omega_m values.

    this avoids calling quad() millions of times inside the MCMC
    sampler. during sampling, integrals are retrieved via
    interpolate_chi().

    parameters
    ----------
    z_arr   : array-like — redshifts of all supernovae
    Om_grid : array-like — Omega_m values to cache

    returns
    -------
    dict mapping Om -> numpy.ndarray of comoving integrals
    """
    cache = {}
    for Om in Om_grid:
        cache[Om] = comoving_distance(z_arr, Om)
    return cache


def interpolate_chi(Om, Om_cache, chi_cache):
    """
    fast linear interpolation of comoving integrals.

    parameters
    ----------
    Om        : float          — query Omega_m value
    Om_cache  : numpy.ndarray  — cached Omega_m grid
    chi_cache : numpy.ndarray  — cached integrals, shape (len(Om_cache), N_sn)

    returns
    -------
    numpy.ndarray — interpolated comoving integrals for all SNe
    """
    k = np.searchsorted(Om_cache, Om)
    k = np.clip(k, 1, len(Om_cache) - 1)
    t = (Om - Om_cache[k-1]) / (Om_cache[k] - Om_cache[k-1])
    return (1.0 - t) * chi_cache[k-1] + t * chi_cache[k]