
---

### `fig4.py`
```python
#!/usr/bin/env python3
"""
fig4.py

Generate Fig. 4: the (ω_on, Pe) stability map showing:
  - λ_real = 0 contour (black solid)
  - oscillatory λ_im > 0 region (red dashed contour & shading)
  - analytic Pe_c(ω_on) line (magenta dashed)
  - phase‐diagram points from CSVs
Saves as `fig4.png` (600 dpi).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
# Publication‐style fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'

# Plot settings
FIG_NAME   = 'fig4.png'
FONTSIZE   = 26
MARKERSIZE = 9
CMAP       = plt.cm.RdBu
NGRID      = 500

# Physical constants
L      = 2 * np.pi
A      = 0.2
Q_CONST= 1.0           # fixed q for stability eval
D      = 0.1
RHO0   = 1.0
WO     = 1.0           # off‐rate
ALPHA  = 10.0

# Phase‐diagram data directory (adjust to your folder)
DATA_DIR_PHASE = Path.home() / 'my_work/active_fluids/numerics/one_dimension/phase_diagram'

# CSV files & marker styles
PHASE_FILES = {
    'Homogeneous':            ('phase_data_homogeneous.csv',          'o', 'k'),
    'Stationary In‑Phase':    ('phase_data_stationary_in_phase.csv',  's', 'darkblue'),
    'Stationary Out‑Phase':   ('phase_data_stationary_out_of_phase.csv','^','darkgreen'),
    'Moving':                 ('phase_data_moving.csv',               'o', 'darkred'),
}
# ──────────────────────────────────────────────────────────────────────────────


def critical_Pe(won: float) -> float:
    """
    Analytic critical Pe from Eq. (10):
      Pe_c = ((1+q_c^2)/((1+αωₒff)ψ₀ fψ))*(1 + D + (ωₒff+ωₒn)/q_c^2)
    """
    if won < 1e-6:
        return np.nan
    psi0 = (won / (WO + won)) * RHO0
    fpsi = 1 / (1 + psi0)**2
    q_c  = ((WO + won) / (1 + D))**0.25
    return ((1 + q_c**2) / ((1 + ALPHA * WO) * psi0 * fpsi)
            * (1 + D + (WO + won) / q_c**2))


def stability_eigen(won: float, pe: float):
    """
    Returns (λ_re, λ_im) for fixed q=Q_CONST, ω_on=won, Pe=pe.
    """
    psi0 = (won / (WO + won)) * RHO0
    phi0 = (WO  / (WO + won)) * RHO0
    B    = pe * (1 + ALPHA * WO) * (psi0 / (1 + psi0)**2)

    m11 = -Q_CONST**2 * (1 - B/(1 + Q_CONST**2)) - WO
    m22 = -Q_CONST**2 * D        - won
    m12 = won
    m21 = (-Q_CONST**2 * (-(pe * (phi0 - ALPHA*WO*psi0)
              / (1 + psi0)**2)) / (1 + Q_CONST**2)) + WO

    tr    = m11 + m22
    det   = m11*m22 - m12*m21
    Δ     = tr**2 - 4 * det

    λ_re = np.where(Δ < 0, tr/2, (tr + np.sqrt(Δ)) / 2)
    λ_im = np.where(Δ < 0, np.sqrt(-Δ)/2, 0.0)
    return λ_re, λ_im


def load_phase_data(filename: str):
    """Load (won, Pe) from a phase‐diagram CSV."""
    df = pd.read_csv(DATA_DIR_PHASE / filename)
    return df['won'].values, df['Pe'].values


def main():
    # Prepare grid
    won_vals = np.linspace(0, 1.1, NGRID)
    pe_vals  = np.linspace(2.0, 3.75, NGRID)
    W, P     = np.meshgrid(won_vals, pe_vals, indexing='xy')

    # Compute λ_re, λ_im on grid
    λ_re, λ_im = stability_eigen(W, P)

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(labelsize=FONTSIZE)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(2.25, 3.55)

    # Contours and shading
    ax.contour( W, P, λ_re, levels=[0], colors='k', linewidths=3 )
    ax.contour( W, P, np.ma.masked_where(λ_re <= 0, λ_im),
                levels=[0], colors='darkred', linewidths=3, linestyles='dashed' )

    ax.contourf(W, P, λ_re <= 0,  levels=[0.5,1], colors=['k'],    alpha=0.1 )
    ax.contourf(W, P, (λ_re > 0) & (λ_im > 0), levels=[0.5,1],
                colors=['darkred'], alpha=0.1 )
    ax.contourf(W, P, λ_re > 0,   levels=[0.5,1], colors=['darkblue'], alpha=0.1 )

    # Analytic critical Pe
    won_line, pe_line = [], []
    for w in won_vals:
        pc = critical_Pe(w)
        if not np.isnan(pc) and np.interp(w, W.flatten(), λ_im.flatten())>0:
            won_line.append(w)
            pe_line.append(pc)
    ax.plot(won_line, pe_line, '--', color='magenta', linewidth=3,
            label=r'$Pe_c$ (Eq.~10)')

    # Overlay phase points
    for label, (fname, marker, col) in PHASE_FILES.items():
        won_d, pe_d = load_phase_data(fname)
        ax.plot(won_d, pe_d, marker=marker, color=col,
                markerfacecolor='none', ms=MARKERSIZE,
                label=label)

    # Labels
    ax.set_xlabel(r'$\omega_{\mathrm{on}}$', fontsize=FONTSIZE, labelpad=-5)
    ax.set_ylabel(r'$Pe$',                 fontsize=FONTSIZE, labelpad=-5)
    ax.legend(loc='upper right', fontsize=FONTSIZE-5)

    plt.tight_layout()
    fig.savefig(FIG_NAME, dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
