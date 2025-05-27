
---

### `fig1c.py`
```python
#!/usr/bin/env python3
"""
fig1c.py

Compute the wavenumber q that maximizes the real part of the growth rate
λ(q) over a grid of ω_on and Pe, then plot the resulting heatmap as panel (c).
Saves figure as `fig1c.png` (600 dpi).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Publication‐style fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'

# Plotting parameters
FONTSIZE = 22
LABELSIZE = 18

# Physical constants
L      = 2 * np.pi      # system size
A      = 0.2            # microscopic cutoff
Q_MIN  = 2 * np.pi / L
Q_MAX  = 2 * np.pi / A
RHO0   = 1.0
WO     = 1.0            # omega_off
ALPHA  = 10.0
D      = 0.1


def growth_rate_real_q(q, won, pe):
    """
    Real part of λ(q) for given wavenumber q, on-rate won, and Pe.
    """
    psi0 = (won / (WO + won)) * RHO0
    phi0 = (WO  / (WO + won)) * RHO0
    B    = pe * (1 + ALPHA * WO) * (psi0 / (1 + psi0)**2)

    m11 = -q**2 * (1 - B/(1 + q**2)) - WO
    m22 = -q**2 * D         - won
    m12 = won
    m21 = -q**2 * (-(pe * (phi0 - ALPHA*WO*psi0) / (1 + psi0)**2)
                   / (1 + q**2)) + WO

    tr    = m11 + m22
    det   = m11*m22 - m12*m21
    delta = tr**2 - 4 * det

    if delta < 0:
        return tr / 2.0
    return (tr + np.sqrt(delta)) / 2.0


def compute_qmax_array(won_vals, pe_vals, q_vals):
    """
    Returns a 2D array where each entry [i,j] is the q maximizing
    growth_rate_real_q(q, won_vals[j], pe_vals[i]).
    """
    qmax = np.zeros((len(pe_vals), len(won_vals)))
    for i, pe in enumerate(pe_vals):
        for j, won in enumerate(won_vals):
            lam = [growth_rate_real_q(q, won, pe) for q in q_vals]
            qmax[i, j] = q_vals[np.argmax(lam)]
    return qmax


def plot_qmax_map(won_vals, pe_vals, qmax_arr):
    """
    Plot filled contour of qmax_arr over (won_vals, pe_vals) grid.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.contourf(won_vals, pe_vals, qmax_arr,
                       levels=200, cmap='RdBu')
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(which='both', direction='in', labelsize=LABELSIZE)
    cbar.set_label(r'$q_{\max}$', fontsize=FONTSIZE)

    ax.set_xlabel(r'$\omega_{\mathrm{on}}$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$Pe$',                 fontsize=FONTSIZE)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', direction='in', labelsize=LABELSIZE)

    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes,
            fontsize=FONTSIZE, va='top', ha='left')

    plt.tight_layout()
    fig.savefig('fig1c.png', dpi=600)
    plt.show()


def main():
    # Parameter grids
    won_vals = np.linspace(0.0, 1.1, 200)
    pe_vals  = np.linspace(1.98, 3.75, 200)
    q_vals   = np.linspace(Q_MIN, Q_MAX, 1000)

    # Compute and plot
    qmax_arr = compute_qmax_array(won_vals, pe_vals, q_vals)
    plot_qmax_map(won_vals, pe_vals, qmax_arr)


if __name__ == '__main__':
    main()
