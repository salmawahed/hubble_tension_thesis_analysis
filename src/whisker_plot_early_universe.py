import matplotlib.pyplot as plt
import numpy as np

# data parsed for early universe (redundancy corrected)
probes = [
    {
        "name": "Planck 2018 (TT, TE, EE + lowE)", 
        "val": 67.27, "err_plus": 0.60, "err_minus": 0.60, 
        "lbl": r"$67.27 \pm 0.60$", "color": "#1f77b4"
    },
    {
        "name": "Planck 2018 (TT, TE, EE + lowE + lensing)", 
        "val": 67.36, "err_plus": 0.54, "err_minus": 0.54, 
        "lbl": r"$67.36 \pm 0.54$", "color": "#2ca02c"
    },
    {
        "name": "Planck 2018 + BAO", 
        "val": 67.66, "err_plus": 0.42, "err_minus": 0.42, 
        "lbl": r"$67.66 \pm 0.42$", "color": "#d62728"
    }
]

# set up clean canvas layout
fig, ax = plt.subplots(figsize=(10, 5))

# vertical distribution of markers
y_positions = np.arange(len(probes))[::-1]

for y, probe in zip(y_positions, probes):
    xerr = [[probe["err_minus"]], [probe["err_plus"]]]
    
    # render error bar tracks without vertical caps
    ax.errorbar(probe["val"], y, xerr=xerr, fmt='o', color=probe["color"],
                mec=probe["color"], mfc=probe["color"], ms=9, lw=3.5, capsize=0)
    
    ax.text(probe["val"], y + 0.18, probe["lbl"], color=probe["color"],
            fontsize=12, ha='center', va='bottom', weight='bold')
    
    # place collaboration identifiers directly below the markers
    ax.text(probe["val"], y - 0.18, probe["name"], color='#444444',
            fontsize=11, ha='center', va='top')

# tightened horizontal limits to clearly show high-precision CMB boundaries
ax.set_xlim(65.5, 69.5)
ax.set_ylim(-0.6, len(probes) - 0.4)
ax.set_xticks(np.arange(66.0, 69.5, 0.5))

# label and tick configuration
ax.tick_params(axis='x', labelsize=14)
ax.set_xlabel(r'$H_0 \ [\mathrm{km \ s}^{-1} \ \mathrm{Mpc}^{-1}]$', fontsize=18, labelpad=12)

# wipe out irrelevant vertical axis lines
ax.set_yticks([])

# dataset category bounding box
ax.text(69.2, len(probes) - 0.6, "Early Universe", fontsize=14,
        bbox=dict(facecolor='white', edgecolor='#333333', boxstyle='square,pad=0.4'))

plt.tight_layout()
plt.savefig('early_universe_h0.png', dpi=600)