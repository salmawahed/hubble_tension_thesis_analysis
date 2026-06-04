import matplotlib.pyplot as plt
import numpy as np

# 1. early universe data array
early_probes = [
    {"name": "Planck 2018 (TT, TE, EE + lowE)", "val": 67.27, "err_plus": 0.60, "err_minus": 0.60, "lbl": r"$67.27 \pm 0.60$", "color": "#cc7a11"},
    {"name": "Planck 2018 (TT, TE, EE + lowE + lensing)", "val": 67.36, "err_plus": 0.54, "err_minus": 0.54, "lbl": r"$67.36 \pm 0.54$", "color": "#b85c00"},
    {"name": "Planck 2018 + BAO", "val": 67.66, "err_plus": 0.42, "err_minus": 0.42, "lbl": r"$67.66 \pm 0.42$", "color": "#cc112c"}
]

# 2. late universe data array
late_probes = [
    {"name": "SH0ES (Riess et al. 2022)", "val": 73.04, "err_plus": 1.04, "err_minus": 1.04, "lbl": r"$73.04 \pm 1.04$", "color": "#3266d6"},
    {"name": "CCHP (Freedman et al. 2024)", "val": 69.96, "err_plus": 1.05, "err_minus": 1.05, "lbl": r"$69.96 \pm 1.05$", "color": "#cc112c"},
    {"name": "H0LiCOW (Wong et al. 2020)", "val": 73.3, "err_plus": 1.7, "err_minus": 1.8, "lbl": r"$73.3^{+1.7}_{-1.8}$", "color": "#1cb82c"},
    {"name": "TDCOSMO (2024)", "val": 74.5, "err_plus": 5.6, "err_minus": 6.1, "lbl": r"$74.5^{+5.6}_{-6.1}$", "color": "#7a117a"},
    {"name": "Abbott et al. (2017)", "val": 70.0, "err_plus": 12.0, "err_minus": 8.0, "lbl": r"$70^{+12}_{-8}$", "color": "#e67e22"}
]

# 3. combined / tension summary data array 
summary_probes = [
    {"name": "combining all", "val": 73.3, "err_plus": 0.8, "err_minus": 0.8, "lbl": r"$73.3 \pm 0.8$", "color": "black", "sigma": r"6.1$\sigma$"},
    {"name": "with Cepheids", "val": 73.9, "err_plus": 1.0, "err_minus": 1.0, "lbl": r"$73.9 \pm 1.0$", "color": "#3266d6", "sigma": r"5.8$\sigma$"},
    {"name": "with TRGB", "val": 72.5, "err_plus": 1.2, "err_minus": 1.2, "lbl": r"$72.5 \pm 1.2$", "color": "#e35185", "sigma": r"4.0$\sigma$"}
]

# recalculated figure height for the new row count
fig, ax = plt.subplots(figsize=(11, 11.5))
y = 12.5

# --- section 1: early universe ---
ax.text(59, y + 0.3, "Early", fontsize=16, weight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
for p in early_probes:
    xerr = [[p["err_minus"]], [p["err_plus"]]]
    ax.errorbar(p["val"], y, xerr=xerr, fmt='o', color=p["color"], ms=9, lw=3.5, capsize=0)
    ax.text(p["val"], y + 0.16, p["lbl"], color=p["color"], fontsize=12, ha='center', va='bottom', weight='bold')
    ax.text(p["val"], y - 0.16, p["name"], color='#555555', fontsize=10, ha='center', va='top')
    y -= 1.3

# section break line
ax.axhline(y=y + 0.5, color='#cccccc', linestyle='--', lw=1.5)
y -= 0.2

# --- section 2: late universe ---
ax.text(82.5, y + 0.2, "Late", fontsize=16, weight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
for p in late_probes:
    xerr = [[p["err_minus"]], [p["err_plus"]]]
    ax.errorbar(p["val"], y, xerr=xerr, fmt='o', color=p["color"], ms=9, lw=3.5, capsize=0)
    ax.text(p["val"], y + 0.16, p["lbl"], color=p["color"], fontsize=12, ha='center', va='bottom', weight='bold')
    ax.text(p["val"], y - 0.16, p["name"], color='#555555', fontsize=10, ha='center', va='top')
    y -= 1.3

# section break line
ax.axhline(y=y + 0.5, color='#cccccc', linestyle='--', lw=1.5)
y -= 0.4

# --- section 3: tension analysis ---
ax.text(81, y + 0.4, "Early vs. Late", fontsize=12, color='#444444', ha='center', weight='bold')
for p in summary_probes:
    xerr = [[p["err_minus"]], [p["err_plus"]]]
    ax.errorbar(p["val"], y, xerr=xerr, fmt='o', color=p["color"], ms=9, lw=3.5, capsize=0)
    
    lbl_text = p["lbl"] + " " + p["name"]
    ax.text(p["val"] + p["err_plus"] + 0.3, y, lbl_text, color=p["color"], fontsize=12, ha='left', va='center')
    ax.text(81, y, p["sigma"], color='#333333', fontsize=13, ha='center', va='center')
    y -= 1.1

ax.set_title(r"flat $-\ \Lambda$CDM", fontsize=24, pad=20, fontname='DejaVu Serif')
ax.set_xlim(57, 85)
ax.set_ylim(y + 0.5, 13.8)
ax.set_xticks(np.arange(58, 86, 2))
ax.tick_params(axis='x', labelsize=16)
ax.set_xlabel(r'$H_0 \ [\mathrm{km \ s}^{-1} \ \mathrm{Mpc}^{-1}]$', fontsize=20, labelpad=15)

ax.set_yticks([])

plt.tight_layout()
plt.savefig('combined_h0_tension_plot_final.png', dpi=600)
plt.show()