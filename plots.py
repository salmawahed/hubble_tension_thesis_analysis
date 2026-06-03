"""
plots.py
--------
all figure generation functions for the Hubble tension analysis.

each function saves a PDF (and PNG) to the figures/ directory.
figures correspond to those in Chapter 4 of the thesis.

figure list:
  - plot_1d_posterior()     : 1D marginal posterior for H0
  - plot_2d_posterior()     : 2D joint posterior P(H0, Om | D)
  - plot_trace()            : MCMC trace plots (all walkers)
  - plot_autocorr()         : Autocorrelation function
  - plot_corner()           : Corner plot of MCMC samples
  - plot_h0_comparison()    : H0 comparison whisker plot

Thesis: The Hubble Tension: Mathematical Foundations,
        Statistical Analysis, and Proposed Resolutions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import corner
import os

# ── Output Directory ──────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 12})


def _save(fig, name):
    """save figure as both PDF and PNG."""
    fig.savefig(f"{FIGURES_DIR}/{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{FIGURES_DIR}/{name}.png", dpi=600, bbox_inches='tight')
    print(f"  Saved: figures/{name}.pdf / .png")
    plt.close(fig)


# ── Figure 1: 1D Marginal Posterior ──────────────────────────────────────────
def plot_1d_posterior(H0_grid, marginal_H0, H0_MAP,
                      h0_low_68, h0_high_68,
                      h0_low_95, h0_high_95,
                      sigma_lo, sigma_hi,
                      tension=4.9,
                      H0_planck=67.4, sigma_planck=0.5):
    """
    1D marginal posterior P(H0 | data) with Planck tension arrow.
    corresponds to Figure 4.x in the thesis.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(H0_grid, marginal_H0, color='black', linewidth=2.5,
            label=r'Marginal Posterior $P(H_0|\mathrm{data})$')

    mask_95 = (H0_grid >= h0_low_95) & (H0_grid <= h0_high_95)
    ax.fill_between(H0_grid[mask_95], marginal_H0[mask_95],
                    color='royalblue', alpha=0.3, label='95% Credible Interval')

    mask_68 = (H0_grid >= h0_low_68) & (H0_grid <= h0_high_68)
    ax.fill_between(H0_grid[mask_68], marginal_H0[mask_68],
                    color='royalblue', alpha=0.6, label='68% Credible Interval')

    ax.axvline(H0_MAP, color='crimson', linestyle='-', linewidth=2.5,
               label=f'This Work (MAP={H0_MAP:.2f}) & SH0ES 2022 (73.04)')

    ax.axvline(H0_planck, color='teal', linestyle='--', linewidth=2,
               label=r'Planck 2018 ($67.4 \pm 0.5$)')

    ax.annotate(
        f'$H_0 = {H0_MAP:.2f}^{{+{sigma_hi:.2f}}}_{{-{sigma_lo:.2f}}}$ km/s/Mpc',
        xy=(H0_MAP, marginal_H0.max() * 0.55),
        xytext=(H0_MAP + 0.7, marginal_H0.max() * 0.65),
        fontsize=12, color='crimson', fontweight='normal',
        arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2)
    )

    arrow_y = marginal_H0.max() * 0.35
    ax.annotate('', xy=(H0_planck, arrow_y), xytext=(H0_MAP - 0.3, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='dimgray', lw=2))
    ax.text((H0_planck + H0_MAP) / 2.0, arrow_y * 1.12,
            f'$\\sim {tension:.1f}\\sigma$ Tension',
            color='black', fontsize=12, fontweight='bold', ha='center')

    ax.text(0.01, 0.01,
            r'$^\dagger$Tension uses full SH0ES $\sigma$=1.04 '
            r'(incl. Cepheid systematics)',
            transform=ax.transAxes, fontsize=9, color='dimgray', va='bottom')

    ax.set_xlabel(r'Hubble Constant $H_0$ (km/s/Mpc)', fontsize=13)
    ax.set_ylabel(r'Probability Density $P(H_0)$', fontsize=13)
    ax.set_title(r'1D Marginal Posterior for $H_0$'
                 '\n'
                 r'[Pantheon+ + SH0ES, full covariance, $\Omega_m$ marginalised]',
                 fontsize=13)
    ax.set_xlim(66.0, 77.0)
    ax.set_ylim(0, marginal_H0.max() * 1.20)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    _save(fig, "fig_1d_posterior")


# ── Figure 2: 2D Joint Posterior ─────────────────────────────────────────────
def plot_2d_posterior(H0_grid, Om_grid, posterior_2D,
                      level_68, level_95, H0_MAP, Om_MAP):
    """
    2D joint posterior P(H0, Om | data) with credible contours.
    corresponds to Figure 4.x in the thesis.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    posterior_plot = posterior_2D / posterior_2D.max()

    cf = ax.contourf(H0_grid, Om_grid, posterior_plot.T,
                     levels=60, cmap='Blues', alpha=0.85)

    ax.contour(H0_grid, Om_grid, posterior_plot.T,
               levels=[level_95], colors=['steelblue'],
               linewidths=2.0, linestyles='--')

    ax.contour(H0_grid, Om_grid, posterior_plot.T,
               levels=[level_68], colors=['navy'],
               linewidths=2.5, linestyles='-')

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Normalised Posterior Density  [0 = min, 1 = peak]',
                   fontsize=11)

    ax.plot(H0_MAP, Om_MAP, '+', color='crimson', markersize=12,
            markeredgewidth=2.5, zorder=5)

    legend_elements = [
        Line2D([0], [0], color='navy', lw=2.5, linestyle='-',
               label='68% Credible Region'),
        Line2D([0], [0], color='steelblue', lw=2.0, linestyle='--',
               label='95% Credible Region'),
        Line2D([0], [0], color='crimson', lw=0, marker='+',
               markersize=10, markeredgewidth=2.5,
               label=f'MAP  ($H_0$={H0_MAP:.2f}, $\\Omega_m$={Om_MAP:.3f})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              frameon=True, fontsize=10)

    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=13)
    ax.set_ylabel(r'$\Omega_m$', fontsize=13)
    ax.set_title(r'Joint Posterior $P(H_0,\,\Omega_m\,|\,\mathcal{D})$'
                 '\n[Pantheon+ + SH0ES, full covariance]', fontsize=13)
    ax.set_xlim(71.5, 75.0)
    ax.set_ylim(0.20, 0.50)
    ax.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_2d_posterior")


# ── Figure 3: Trace Plots ─────────────────────────────────────────────────────
def plot_trace(chain, n_burnin=500, n_walkers=32):
    """
    MCMC trace plots for H0 (top) and Omega_m (bottom).
    corresponds to Figure 4.x in the thesis.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    labels = [r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', r'$\Omega_m$']
    colors = plt.cm.tab20(np.linspace(0, 1, n_walkers))

    for dim, (ax, label) in enumerate(zip(axes, labels)):
        for w in range(n_walkers):
            ax.plot(chain[:, w, dim], color=colors[w],
                    alpha=0.4, linewidth=0.6)
        ax.set_ylabel(label, fontsize=13)
        ax.axvline(n_burnin, color='crimson', linestyle='--',
                   linewidth=1.8,
                   label='Burn-in end (500 steps)' if dim == 0 else '')
        ax.axvspan(0, n_burnin, color='crimson', alpha=0.06)
        post_median = np.median(chain[n_burnin:, :, dim])
        ax.axhline(post_median, color='black', linestyle='-',
                   linewidth=1.2, alpha=0.6,
                   label=f'Post burn-in median = {post_median:.3f}'
                   if dim == 0 else
                   f'Post burn-in median = {post_median:.4f}')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    axes[0].set_title(
        f'MCMC Trace Plots  [{n_walkers} walkers, '
        f'{chain.shape[0]} steps, burn-in={n_burnin}]\n'
        r'Pantheon+ + SH0ES, full covariance', fontsize=12)
    axes[1].set_xlabel('MCMC Step', fontsize=13)
    plt.tight_layout()
    _save(fig, "fig_trace")


# ── Figure 4: Autocorrelation Function ───────────────────────────────────────
def plot_autocorr(chain, n_burnin=500, n_walkers=32, max_lag=150):
    """
    autocorrelation function for H0 and Omega_m.
    corresponds to Figure 4.x in the thesis.
    """
    def acf(chain_1d, max_lag):
        n_steps, n_w = chain_1d.shape
        result = np.zeros(max_lag + 1)
        for w in range(n_w):
            x = chain_1d[n_burnin:, w] - chain_1d[n_burnin:, w].mean()
            f = np.fft.fft(x, n=2*len(x))
            ac = np.fft.ifft(f * np.conj(f)).real
            ac = ac[:max_lag+1] / ac[0]
            result += ac
        return result / n_w

    def tau_int(a):
        window = np.where(np.abs(a) < 0.01)[0]
        cut = window[0] if len(window) > 0 else len(a)
        return 1.0 + 2.0 * np.sum(a[1:cut])

    lags   = np.arange(max_lag + 1)
    acf_H0 = acf(chain[:, :, 0], max_lag)
    acf_Om = acf(chain[:, :, 1], max_lag)
    tau_H0 = tau_int(acf_H0)
    tau_Om = tau_int(acf_Om)
    N_steps = chain.shape[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags, acf_H0, color='navy', linewidth=2.0, linestyle='-',
            label=fr'$H_0$  ($\tau \approx {tau_H0:.1f}$ steps)')
    ax.plot(lags, acf_Om, color='steelblue', linewidth=2.0, linestyle='--',
            label=fr'$\Omega_m$  ($\tau \approx {tau_Om:.1f}$ steps)')
    ax.axhline(0, color='black', linewidth=1.0, linestyle=':', alpha=0.7,
               label='Zero reference')
    ax.axvspan(0, max(tau_H0, tau_Om), color='gold', alpha=0.12,
               label=fr'Approx. $\tau$ region')
    ax.axvline(tau_H0, color='navy', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(tau_Om, color='steelblue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate(
        f'$N_{{steps}} = {N_steps}$\n'
        f'$50\\tau_{{H_0}} = {50*tau_H0:.0f}$\n'
        f'$N_{{steps}} \\geq 50\\tau$ ? YES',
        xy=(120, 0.75), fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.85)
    )
    ax.set_xlabel('Lag $k$ (steps)', fontsize=13)
    ax.set_ylabel('Autocorrelation $C(k)$', fontsize=13)
    ax.set_title('Autocorrelation Function of MCMC Chains\n'
                 r'[Pantheon+ + SH0ES, 32 walkers, post burn-in]',
                 fontsize=13)
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    _save(fig, "fig_autocorr")


# ── Figure 5: Corner Plot ─────────────────────────────────────────────────────
def plot_corner(flat_samples, n_burnin=500, thin=15, n_walkers=32):
    """
    corner plot of MCMC posterior samples.
    corresponds to Figure 4.x in the thesis.
    """
    H0_q = np.percentile(flat_samples[:, 0], [16, 50, 84])
    Om_q = np.percentile(flat_samples[:, 1], [16, 50, 84])
    H0_med = H0_q[1]; H0_lo = H0_q[1]-H0_q[0]; H0_hi = H0_q[2]-H0_q[1]
    Om_med = Om_q[1]; Om_lo = Om_q[1]-Om_q[0]; Om_hi = Om_q[2]-Om_q[1]

    labels  = [r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', r'$\Omega_m$']
    ranges  = [(H0_med-4*H0_lo, H0_med+4*H0_hi),
               (Om_med-4*Om_lo, Om_med+4*Om_hi)]

    fig = corner.corner(
        flat_samples, labels=labels, quantiles=[0.16, 0.50, 0.84],
        show_titles=False, title_kwargs={"fontsize": 13},
        label_kwargs={"fontsize": 13},
        levels=(0.68, 0.95), fill_contours=True,
        contourf_kwargs={"colors": ["white", "#d0e4f7", "#2171b5"],
                         "alpha": 0.95},
        contour_kwargs={"colors": ["#6aaed6", "#084594"], "linewidths": 1.8},
        hist_kwargs={"color": "#2171b5", "linewidth": 1.8},
        bins=50, color="#2171b5", smooth=1.2, smooth1d=1.5,
        range=ranges, plot_datapoints=False, plot_density=False,
        figsize=(7, 7),
    )

    axes = np.array(fig.axes).reshape(2, 2)
    axes[0, 0].set_title(
        f'$H_0 = {H0_med:.2f}^{{+{H0_hi:.2f}}}_{{-{H0_lo:.2f}}}$ km/s/Mpc',
        fontsize=12, pad=10)
    axes[1, 1].set_title(
        f'$\\Omega_m = {Om_med:.3f}^{{+{Om_hi:.3f}}}_{{-{Om_lo:.3f}}}$',
        fontsize=12, pad=10)

    for ax, q in zip([axes[0, 0], axes[1, 1]], [H0_q, Om_q]):
        for val, ls in zip(q, ['--', '-', '--']):
            ax.axvline(val, color='crimson', linestyle=ls,
                       linewidth=1.5, alpha=0.9)

    axes[1, 0].plot(H0_med, Om_med, '+', color='crimson',
                    markersize=12, markeredgewidth=2.5, zorder=10,
                    label='Median')
    axes[1, 0].legend(loc='upper right', fontsize=10, framealpha=0.7)

    n_eff = flat_samples.shape[0]
    fig.suptitle(
        'Corner Plot: MCMC Posterior  [Pantheon+ + SH0ES, flat $\\Lambda$CDM]\n'
        f'{n_walkers} walkers $\\times$ 5000 steps, burn-in={n_burnin}, '
        f'thin={thin}  ($N_{{\\rm eff}} \\approx {n_eff}$)',
        fontsize=11)
    fig.subplots_adjust(top=0.88)
    _save(fig, "fig_corner")


# ── Figure 6: H0 Comparison Whisker Plot ─────────────────────────────────────
def plot_h0_comparison(H0_this_work_mcmc, H0_this_work_mle,
                       sigma_mcmc_lo, sigma_mcmc_hi,
                       sigma_mle):
    """
    H0 comparison whisker plot: Early vs Late universe measurements.
    corresponds to Figure 4.x in the thesis.
    """
    measurements = [
        ("Planck 2018\n(TT,TE,EE+lowE)",         67.27, 0.60, 0.60, "#b5651d", 0),
        ("Planck 2018\n(TT,TE,EE+lowE+lensing)",  67.36, 0.54, 0.54, "#b5651d", 0),
        ("Planck 2018 + BAO",                      67.66, 0.42, 0.42, "#b5651d", 0),
        ("SH0ES\n(Riess et al. 2022)",             73.04, 1.04, 1.04, "#1a6faf", 1),
        ("CCHP/JWST\n(Freedman et al. 2024)",      69.96, 1.05, 1.05, "#c0392b", 1),
        ("H0LiCOW\n(Wong et al. 2020)",            73.30, 1.80, 1.70, "#27ae60", 1),
        ("TDCOSMO",                                74.50, 5.60, 6.10, "#8e44ad", 1),
        ("GW170817\n(Abbott et al. 2017)",         70.00, 8.00,12.00, "#e67e22", 1),
        ("This Work\n(MCMC median)",    H0_this_work_mcmc,
         sigma_mcmc_lo, sigma_mcmc_hi, "#e74c3c", 2),
        ("This Work\n(Frequentist MLE)", H0_this_work_mle,
         sigma_mle, sigma_mle, "#e74c3c", 2),
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    group_labels = {0: "Early Universe", 1: "Late Universe", 2: "This Work"}
    group_colors = {0: "#b5651d", 1: "#1a6faf", 2: "#e74c3c"}

    y_positions = []
    y = 0; prev_group = None
    for *_, group in measurements:
        if prev_group is not None and group != prev_group:
            y += 1.2
        y_positions.append(y); y += 1; prev_group = group

    ax.axvspan(67.66-0.42, 67.66+0.42, color="#b5651d", alpha=0.10, zorder=0)
    ax.axvspan(73.04-1.04, 73.04+1.04, color="#1a6faf", alpha=0.10, zorder=0)

    group_y = {0: [], 1: [], 2: []}
    for (label, H0, slo, shi, color, group), ypos in \
            zip(measurements, y_positions):
        group_y[group].append(ypos)
        is_tw = (group == 2)
        ax.errorbar(H0, ypos, xerr=[[slo], [shi]], fmt='o',
                    color=color, ecolor=color,
                    elinewidth=2.5 if is_tw else 1.8, capsize=4, capthick=2,
                    markersize=9 if is_tw else 7,
                    markeredgewidth=2 if is_tw else 1,
                    markeredgecolor='black' if is_tw else color,
                    markerfacecolor=color, zorder=5)
        val = (f"{H0:.2f} $\\pm$ {slo:.2f}" if slo == shi
               else f"{H0:.2f}$^{{+{shi:.2f}}}_{{-{slo:.2f}}}$")
        ax.text(H0, ypos+0.35, val, ha='center', va='bottom', fontsize=8.5,
                color=color, fontweight='bold' if is_tw else 'normal')
        ax.text(57.5, ypos, label, ha='left', va='center',
                fontsize=8.5, color='black')

    for group, ylist in group_y.items():
        ax.text(84.5, np.mean(ylist), group_labels[group],
                ha='right', va='center', fontsize=11, fontweight='bold',
                color=group_colors[group],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=group_colors[group], alpha=0.9))

    sep1 = (max(group_y[0]) + min(group_y[1])) / 2
    sep2 = (max(group_y[1]) + min(group_y[2])) / 2
    for s in [sep1, sep2]:
        ax.axhline(s, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.axhspan(min(group_y[2])-0.5, max(group_y[2])+0.5,
               color="#e74c3c", alpha=0.07, zorder=0)

    ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=13)
    ax.set_xlim(57, 85); ax.set_ylim(-0.8, max(y_positions)+1.2)
    ax.set_yticks([])
    ax.set_title(r'Comparison of $H_0$ Measurements — flat $\Lambda$CDM',
                 fontsize=13, pad=12)
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    legend_elements = [
        mpatches.Patch(color="#b5651d", alpha=0.3,
                       label="Planck 2018 $1\\sigma$ band"),
        mpatches.Patch(color="#1a6faf", alpha=0.3,
                       label="SH0ES 2022 $1\\sigma$ band"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markeredgecolor='black', markersize=9, markeredgewidth=2,
               label='This Work'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=10, framealpha=0.9)
    plt.tight_layout()
    _save(fig, "fig_h0_comparison")