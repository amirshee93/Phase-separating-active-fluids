
---

### `fig3.py`
```python
#!/usr/bin/env python3
"""
fig3.py

Creates a 2×2 panel:
 (a,b) ψ & φ concentration profiles
 (c,d) v, ∂ₓv, ∂ₓ²v profiles

for two parameter sets, and saves as `fig3.png` (600 dpi).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

# ──────────────────────────────────────────────────────────────────────────────
# Publication‐style fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.latex.preamble'] = r'\usepackage[usenames,dvipsnames]{xcolor}'

# Layout & styling
FONTSIZE   = 28
XLABELSIZE = 20
CMAP       = plt.cm.RdBu
FIG_NAME   = 'fig3.png'

# Data locations & parameters (edit as needed)
# Panel (a,c)
DATA_FOLDER_1 = '/Users/amirshee/my_work/active_fluids/numerics/one_dimension/won_0.15/pe_3.5/dynamics_data'
# Panel (b,d)
DATA_FOLDER_2 = '/Users/amirshee/my_work/active_fluids/numerics/one_dimension/won_0.5/pe_3.0/dynamics_data'
FILE_PREFIX   = 'concentrations_velocity_'
FILE_INDEX    = 900  # which time‐slice to load
# ──────────────────────────────────────────────────────────────────────────────


def load_data(folder, prefix, idx):
    """Load x, ψ, φ, v from CSV at folder/prefix{idx}.csv."""
    path = os.path.join(folder, f"{prefix}{idx}.csv")
    df = pd.read_csv(path)
    x   = df['xval'].to_numpy()
    psi = df[' caval'].to_numpy()
    phi = df[' cpval'].to_numpy()
    v   = df[' vval'].to_numpy()
    return x, psi, phi, v


def compute_derivatives(x, v):
    """Return first and second spatial derivatives of v."""
    dv  = np.gradient(v, x)
    d2v = np.gradient(dv, x)
    return dv, d2v


def setup_axes():
    """Create 2×2 grid of axes with shared styling."""
    fig = plt.figure(figsize=(8, 6))
    axs = [
        plt.subplot2grid((2, 2), (r, c))
        for r in (0, 1) for c in (0, 1)
    ]
    for ax in axs:
        ax.tick_params(labelsize=XLABELSIZE)
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which='major', direction='in',
                       bottom=True, top=True,
                       left=True, right=True)
        ax.tick_params(which='minor', direction='in',
                       bottom=True, top=True,
                       left=True, right=True)
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    plt.subplots_adjust(
        left=0.095, right=0.97,
        bottom=0.09, top=0.98,
        wspace=0.25, hspace=0.25
    )
    return fig, axs


def plot_panel(ax, x, psi, phi, v, label):
    """
    For panels a/b: plot psi & phi
    For panels c/d: plot v, dv/dx, d2v/dx2
    label: '(a)'…'(d)'
    """
    ax.text(0.025, 0.95, label, transform=ax.transAxes,
            fontsize=FONTSIZE, fontweight='bold', va='top')

    if label in ('(a)', '(b)'):
        ax.plot(x, psi, color='darkred',   linewidth=4, label=r'$\psi$')
        ax.plot(x, phi, color='darkblue',  linewidth=2, label=r'$\phi$')
        ax.fill_between(x, 0, psi, color='darkred',  alpha=0.1)
        ax.fill_between(x, 0, phi, color='darkblue', alpha=0.1)
        ax.set_ylim(0, 1.6 if label=='(a)' else 1.25)

    else:
        dv, d2v = compute_derivatives(x, v)
        ax.plot(x, v,   color='k',         linewidth=2.5, label='v')
        ax.plot(x, dv,  color='darkviolet',linewidth=2.5, linestyle='--', label=r'$\sigma_p$')
        ax.plot(x, d2v, color='darkgreen', linewidth=2.5, linestyle='-.', label='$f_p$')
        ax.set_ylim(-0.32, 0.32) if label=='(c)' else ax.set_ylim(-0.6, 0.6)
        ax.set_xlabel(r'$x$', fontsize=FONTSIZE, labelpad=-7)

    # annotate side legends only on left panels
    if label in ('(a)', '(c)'):
        colors = {'(a)':'darkred','(c)':'k'}
        texts = {'(a)':r'$\psi,~$', '(c)':r'$v,~$'}
        ax.text(-0.24, 0.49 if label=='(a)' else 0.35,
                texts[label],
                transform=ax.transAxes,
                fontsize=FONTSIZE,
                fontweight='bold',
                va='top',
                color=colors[label],
                rotation=90)
        if label=='(a)':
            ax.text(-0.24, 0.59, r'$\phi$',
                    transform=ax.transAxes,
                    fontsize=FONTSIZE,
                    fontweight='bold',
                    va='top',
                    color='darkblue',
                    rotation=90)
        else:
            ax.text(-0.25, 0.59, r'$\sigma_p,~$',
                    transform=ax.transAxes,
                    fontsize=FONTSIZE,
                    fontweight='bold',
                    va='top',
                    color='darkviolet',
                    rotation=90)
            ax.text(-0.24, 0.75, r'$f_p$',
                    transform=ax.transAxes,
                    fontsize=FONTSIZE,
                    fontweight='bold',
                    va='top',
                    color='darkgreen',
                    rotation=90)


def main():
    fig, axs = setup_axes()

    # Panels a & c
    x1, psi1, phi1, v1 = load_data(DATA_FOLDER_1, FILE_PREFIX, FILE_INDEX)
    plot_panel(axs[0], x1, psi1, phi1, v1, '(a)')
    plot_panel(axs[2], x1, psi1, phi1, v1, '(c)')

    # Panels b & d
    x2, psi2, phi2, v2 = load_data(DATA_FOLDER_2, FILE_PREFIX, FILE_INDEX)
    plot_panel(axs[1], x2, psi2, phi2, v2, '(b)')
    plot_panel(axs[3], x2, psi2, phi2, v2, '(d)')

    plt.show()
    fig.savefig(FIG_NAME, dpi=600)


if __name__ == '__main__':
    main()
