import matplotlib.pyplot as plt
import numpy as np

# cleanly mapped data array with exact precision strings to prevent rounding bugs
probes = [
    {
        "name": "SH0ES (Riess et al. 2022)", 
        "val": 73.04, "err_plus": 1.04, "err_minus": 1.04, 
        "lbl": r"$73.04 \pm 1.04$", "color": "#3266d6"
    },
    {
        "name": "CCHP (Freedman et al. 2024)", 
        "val": 69.96, "err_plus": 1.05, "err_minus": 1.05, 
        "lbl": r"$69.96 \pm 1.05$", "color": "#cc112c"
    },
    {
        "name": "H0LiCOW (Wong et al. 2020)", 
        "val": 73.3, "err_plus": 1.7, "err_minus": 1.8, 
        "lbl": r"$73.3^{+1.7}_{-1.8}$", "color": "#1cb82c"
    },
    {
        "name": "TDCOSMO (Birrer et al. 2020)", 
        "val": 74.5, "err_plus": 5.6, "err_minus": 6.1, 
        "lbl": r"$74.5^{+5.6}_{-6.1}$", "color": "#7a117a"
    },
    {
        "name": "Abbott et al. (2017)", 
        "val": 70.0, "err_plus": 12.0, "err_minus": 8.0, 
        "lbl": r"$70^{+12}_{-8}$", "color": "#e67e22"
    }
]

# set up canvas layout
fig, ax = plt.subplots(figsize=(10, 7))

# distribute markers down the vertical axis 
y_positions = np.arange(len(probes))[::-1]

for y, probe in zip(y_positions, probes):
    # construct asymmetric error array structure: [[lower], [upper]]
    xerr = [[probe["err_minus"]], [probe["err_plus"]] ]
    
    # render error bar tracks without vertical caps
    ax.errorbar(probe["val"], y, xerr=xerr, fmt='o', color=probe["color"],
                mec=probe["color"], mfc=probe["color"], ms=9, lw=3.5, capsize=0)
    
    # place formatted LaTeX text strings right above the markers
    ax.text(probe["val"], y + 0.18, probe["lbl"], color=probe["color"],
            fontsize=12, ha='center', va='bottom', weight='bold')
    
    # place collaboration identifiers directly below the markers
    ax.text(probe["val"], y - 0.18, probe["name"], color='#444444',
            fontsize=11, ha='center', va='top')

# broadened horizontal limits to gracefully fit Abbott's lower bound (70 - 8 = 62)
ax.set_xlim(58, 86)
ax.set_ylim(-0.8, len(probes) - 0.2)
ax.set_xticks(np.arange(60, 86, 2))

ax.tick_params(axis='x', labelsize=14)
ax.set_xlabel(r'$H_0 \ [\mathrm{km \ s}^{-1} \ \mathrm{Mpc}^{-1}]$', fontsize=18, labelpad=12)

# wipe out irrelevant vertical axis counts
ax.set_yticks([])

# dataset category bounding box
ax.text(84.5, len(probes) - 0.5, "Late Universe", fontsize=14,
        bbox=dict(facecolor='white', edgecolor='#333333', boxstyle='square,pad=0.4'))

plt.tight_layout()
plt.show()