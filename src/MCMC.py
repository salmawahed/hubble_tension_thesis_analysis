"""
mcmc.py
-------
bayesian MCMC analysis using the emcee ensemble sampler.

handles:
  - walker initialization
  - burn-in and thinning
  - convergence diagnostics:
      * trace plots
      * autocorrelation time tau
      * Gelman-Rubin R-hat statistic
      * effective sample size N_eff
  - posterior summarization (median, 16th/84th percentiles)
  - tension metric vs Planck 2018

this is the computationally expensive module (30-90 min runtime).

Thesis: The Hubble Tension: Mathematical Foundations,
        Statistical Analysis, and Proposed Resolutions
"""

import numpy as np
import emcee
from likelihood import log_posterior

# ── MCMC Configuration ────────────────────────────────────────────────────────
N_WALKERS = 32
N_STEPS   = 5000
N_BURNIN  = 500
THIN      = 15
N_DIM     = 2
SEED      = 42

# starting position: small Gaussian ball around a good initial guess
P0_CENTER  = np.array([73.0, 0.35])
P0_SCATTER = np.array([0.5,  0.02])


def run_sampler(z, mu_obs, cho, const, Om_cache, chi_cache,
                n_walkers=N_WALKERS, n_steps=N_STEPS, seed=SEED,
                progress=True):
    """
    run the emcee EnsembleSampler.

    parameters
    ----------
    z, mu_obs           : data arrays
    cho, const          : cholesky factorization and normalization constant
    Om_cache, chi_cache : cached integrals for interpolation
    n_walkers           : number of walkers (default 32)
    n_steps             : steps per walker (default 5000)
    seed                : random seed for reproducibility (default 42)
    progress            : show tqdm progress bar (default True)

    returns
    -------
    sampler : emcee.EnsembleSampler — completed sampler object
    """
    np.random.seed(seed)
    p0 = P0_CENTER + P0_SCATTER * np.random.randn(n_walkers, N_DIM)

    sampler = emcee.EnsembleSampler(
        n_walkers, N_DIM, log_posterior,
        args=(z, mu_obs, cho, const, Om_cache, chi_cache)
    )
    sampler.run_mcmc(p0, n_steps, progress=progress)
    return sampler


def get_flat_samples(sampler, n_burnin=N_BURNIN, thin=THIN):
    """
    extract flat posterior samples after discarding burn-in and thinning.

    parameters
    ----------
    sampler  : emcee.EnsembleSampler
    n_burnin : steps to discard as burn-in (default 500)
    thin     : thinning factor (default 15)

    returns
    -------
    flat_samples : numpy.ndarray, shape (N_eff, 2) — [H0, Om] samples
    """
    return sampler.get_chain(discard=n_burnin, thin=thin, flat=True)


def autocorr_time(sampler):
    """
    estimate integrated autocorrelation time for each parameter.

    returns
    -------
    tau : numpy.ndarray — [tau_H0, tau_Om]
    """
    try:
        return sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError as e:
        print(f"  Warning: autocorr estimate unreliable: {e}")
        return np.array([np.nan, np.nan])


def gelman_rubin(chain, n_burnin=N_BURNIN):
    """
    compute the Gelman-Rubin R-hat convergence statistic.

    R-hat < 1.01 indicates convergence.

    parameters
    ----------
    chain    : numpy.ndarray, shape (N_steps, N_walkers, N_dim)
    n_burnin : steps to discard before computing R-hat

    returns
    -------
    rhat : numpy.ndarray — R-hat for each parameter
    """
    post  = chain[n_burnin:, :, :]   # (N, M, D)
    N, M, D = post.shape
    rhat = np.empty(D)

    for d in range(D):
        chains = post[:, :, d]         # (N, M)
        W  = np.mean(np.var(chains, axis=0, ddof=1))
        B  = N * np.var(np.mean(chains, axis=0), ddof=1)
        V  = (1.0 - 1.0/N) * W + (1.0/N) * B
        rhat[d] = np.sqrt(V / W)

    return rhat


def posterior_summary(flat_samples):
    """
    compute median and 68% credible interval for each parameter.

    parameters
    ----------
    flat_samples : numpy.ndarray, shape (N_eff, 2)

    returns
    -------
    summary : dict with keys 'H0' and 'Om', each containing
              {'median', 'lo', 'hi', 'q16', 'q84'}
    """
    summary = {}
    names = ['H0', 'Om']
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        summary[name] = {
            'median': q50,
            'lo':     q50 - q16,
            'hi':     q84 - q50,
            'q16':    q16,
            'q84':    q84,
        }
    return summary


def tension_metric(H0_mcmc, sigma_shoes=1.04,
                   H0_planck=67.4, sigma_planck=0.5):
    """
    compute the gaussian tension between this work and Planck 2018.

    T = |H0_this - H0_Planck| / sqrt(sigma_SH0ES^2 + sigma_Planck^2)

    the full SH0ES uncertainty is used (not the tighter SNe-only
    uncertainty) as it represents the complete distance ladder error
    budget and is the standard convention in the literature.

    parameters
    ----------
    H0_mcmc      : float — MCMC median H0
    sigma_shoes  : float — full SH0ES 2022 uncertainty (default 1.04)
    H0_planck    : float — Planck 2018 central value (default 67.4)
    sigma_planck : float — Planck 2018 uncertainty (default 0.5)

    returns
    -------
    tension : float — tension in units of sigma
    """
    delta = abs(H0_mcmc - H0_planck)
    sigma = np.sqrt(sigma_shoes**2 + sigma_planck**2)
    return delta / sigma


def p_planck(H0_samples, H0_planck=67.4):
    """
    bayesian probability P(H0 <= H0_Planck | data).

    the fraction of MCMC samples at or below the Planck central value.
    this is the Bayesian measure of incompatibility.

    parameters
    ----------
    H0_samples : numpy.ndarray — posterior samples for H0
    H0_planck  : float         — Planck central value (default 67.4)

    returns
    -------
    float — posterior probability
    """
    return np.mean(H0_samples <= H0_planck)


def print_mcmc_summary(summary, tau, rhat, n_eff, tension):
    """print a formatted MCMC results summary."""
    print("\n" + "=" * 55)
    print("  MCMC POSTERIOR RESULTS")
    print("=" * 55)
    H0 = summary['H0']
    Om = summary['Om']
    print(f"  H0  median : {H0['median']:.2f} "
          f"+{H0['hi']:.2f} / -{H0['lo']:.2f} km/s/Mpc")
    print(f"  Om  median : {Om['median']:.3f} "
          f"+{Om['hi']:.3f} / -{Om['lo']:.3f}")
    print(f"  tau (H0)   : {tau[0]:.1f} steps")
    print(f"  tau (Om)   : {tau[1]:.1f} steps")
    print(f"  N_eff      : ~{n_eff}")
    print(f"  R-hat (H0) : {rhat[0]:.4f}")
    print(f"  R-hat (Om) : {rhat[1]:.4f}")
    print(f"  Tension    : {tension:.2f} sigma")
    print("=" * 55)


def save_chain(sampler, path_chain="mcmc_chain.npy",
               path_logprob="mcmc_log_prob.npy"):
    """save chain and log-probability arrays to disk."""
    np.save(path_chain,   sampler.get_chain())
    np.save(path_logprob, sampler.get_log_prob())
    print(f"  Chain saved to {path_chain}")
    print(f"  Log-prob saved to {path_logprob}")


def load_chain(path_chain="mcmc_chain.npy"):
    """load a previously saved chain from disk."""
    return np.load(path_chain)
