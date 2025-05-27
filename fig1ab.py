
---

### `fig1ab.py`
```python
#!/usr/bin/env python3
"""
fig1ab.py

Generates a two‑panel stability map:
 (a) λ_im(q, Pe) & contour of λ_re=0
 (b) λ_im(q, ω_on) & contour of λ_re=0

Saves figure as `fig1ab.png` at 600 dpi.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Publication‐style fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'

# Grid and plotting parameters
NGRID    = 1000
FONTSIZE = 22
CMAP     = plt.colormaps.get_cmap('Greys')


def lambda_parts_pe(q, pe, D=0.1, rho0=1.0, wo=1.0, won=1.0, alpha=10.0):
    """Return (λ_re, λ_im) for given q, Pe (Fig a)."""
    psi0 = (won / (wo + won)) * rho0
    phi0 = (wo  / (wo + won)) * rho0
    A    = 1 + D
    B    = pe * (1 + alpha * wo) * (psi0 / (1 + psi0)**2)
    tr   = -q**2 * (A - B/(1 + q**2)) - (wo + won)
    m11  = -q**2 * (1 - B/(1 + q**2)) - wo
    m22  = -q**2 * D - won
    m12  = won
    m21  = -q**2 * (-(pe * (phi0 - alpha*wo*psi0) * (1/(1+psi0)**2)) 
                    / (1+q**2)) + wo
    det  = m11*m22 - m12*m21
    Δ    = tr**2 - 4 * det

    λ_re = np.where(Δ < 0, tr/2, (tr + np.sqrt(Δ)) / 2)
    λ_im = np.where(Δ < 0,  np.sqrt(-Δ)/2, 0.0)
    return λ_re, λ_im


def lambda_parts_won(q, won, D=0.1, rho0=1.0, wo=1.0,
                     pe=2.75, alpha=10.0):
    """Return (λ_re, λ_im) for given q, ω_on (Fig b)."""
    psi0 = (won / (wo + won)) * rho0
    phi0 = (wo  / (wo + won)) * rho0
    A    = 1 + D
    B    = pe * (1 + alpha * wo) * (psi0 / (1 + psi0)**2)
    tr   = -q**2 * (A - B/(1 + q**2)) - (wo + won)
    m11  = -q**2 * (1 - B/(1 + q**2)) - wo
    m22  = -q**2 * D - won
    m12  = won
    m21  = -q**2 * (-(pe * (phi0 - alpha*wo*psi0) * (1/(1+psi0)**2)) 
                    / (1+q**2)) + wo
    det  = m11*m22 - m12*m21
    Δ    = tr**2 - 4 * det

    λ_re = np.where(Δ < 0, tr/2, (tr + np.sqrt(Δ)) / 2)
    λ_im = np.where(Δ < 0,  np.sqrt(-Δ)/2, 0.0)
    return λ_re, λ_im


def make_axes():
    """Set up figure & two subplots with shared styling."""
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=FONTSIZE-5,
                       direction='in', top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax2.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.subplots_adjust(
        left=0.065, right=0.95,
        bottom=0.17, top=0.96,
        wspace=0.35, hspace=0.0
    )
    return fig, ax1, ax2


def plot_panel(ax, q_vals, param_vals, λ_re, λ_im,
               xlabel, ylabel, label_pos, text_a):
    """Draw heatmap, real‑zero contour, and labels on ax."""
    mesh = ax.pcolormesh(q_vals, param_vals, λ_im,
                         shading='auto', cmap=CMAP)
    ax.contour(q_vals, param_vals, λ_im, levels=[0],
               colors='grey', linestyles='dashed')
    ax.contour(q_vals, param_vals, λ_re, levels=[0],
               colors='black')

    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
    cbar.ax.tick_params(direction='in', labelsize=FONTSIZE, size=4)
    cbar.set_label(r'$\lambda_{\mathrm{Im}}$',
                   fontsize=FONTSIZE, labelpad=2)

    ax.set_xlim(q_vals.min(), q_vals.max())
    ax.set_ylim(param_vals.min(), param_vals.max())
    ax.set_xlabel(xlabel, fontsize=FONTSIZE, labelpad=-5)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=2)
    ax.text(*label_pos, text_a,
            transform=ax.transAxes,
            fontsize=FONTSIZE, fontweight='bold', va='top')
    ax.text(0.28, 0.75 if text_a=='(a)' else 0.30,
            r'$\lambda_{\mathrm{Re}}>0$',
            transform=ax.transAxes,
            fontsize=FONTSIZE, fontweight='bold',
            color='red')
    ax.text(0.25 if text_a=='(a)' else 0.50,
            0.50 if text_a=='(a)' else 0.94,
            r'$\lambda_{\mathrm{Re}}<0$',
            transform=ax.transAxes,
            fontsize=FONTSIZE, fontweight='bold',
            color='red')


def main():
    # Prepare grids
    q1 = np.linspace(0, 3, NGRID)
    pe = np.linspace(0, 5, NGRID)
    λr1, λi1 = lambda_parts_pe(*np.meshgrid(q1, pe, indexing='xy'))

    q2   = np.linspace(0, 2, NGRID)
    wonv = np.linspace(0, 2, NGRID)
    λr2, λi2 = lambda_parts_won(*np.meshgrid(q2, wonv, indexing='xy'))

    # Plot
    fig, ax1, ax2 = make_axes()
    plot_panel(ax1, q1, pe,    λr1, λi1, '$q$', '$Pe$',    (0.05, 0.98), '(a)')
    plot_panel(ax2, q2, wonv, λr2, λi2, '$q$', r'$\omega_{\mathrm{on}}$', (0.05, 0.98), '(b)')

    # Save & show
    fig.savefig('fig1ab.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
