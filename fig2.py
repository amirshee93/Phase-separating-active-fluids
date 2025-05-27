
---

### `fig2.py`
```python
#!/usr/bin/env python3
"""
fig2.py

Load a series of CSV files with columns [x, ψ, φ, v] and
produce three stacked kymograph panels (ψ, φ, v) as fig2.png.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
# Plot style
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'

FONTSIZE       = 24
COLORMAP       = plt.cm.RdBu
COLORBAR_WIDTH = 0.02
COLORBAR_PAD   = 0.01

# Data settings (adjust these to your data location)
NUM_FILES   = 1000
FILE_PREFIX = "concentrations_velocity_"
DATA_FOLDER = (
    "../../numerics/one_dimension/won_0.5/pe_2.6/dynamics_data"
)
FILE_EXT    = ".csv"

# Output
FIG_NAME    = "fig2.png"
# ──────────────────────────────────────────────────────────────────────────────


def load_time_series(folder, prefix, n_files, ext):
    """
    Load x, psi, phi, v from files:
      folder/prefix{t}.csv  for t in [0..n_files-1].
    Returns (x, psi_arr, phi_arr, v_arr) as 2D arrays.
    """
    psi_list = []
    phi_list = []
    vel_list = []
    x_vals   = None

    for t in range(n_files):
        path = os.path.join(folder, f"{prefix}{t}{ext}")
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        if x_vals is None:
            x_vals = data[:, 0]
        psi_list.append(data[:, 1])
        phi_list.append(data[:, 2])
        vel_list.append(data[:, 3])

    psi_arr = np.array(psi_list).T
    phi_arr = np.array(phi_list).T
    vel_arr = np.array(vel_list).T
    return x_vals, psi_arr, phi_arr, vel_arr


def plot_kymographs(x, psi, phi, vel):
    """
    Plot three stacked kymographs for psi, phi, and vel.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(
        left=0.08, right=0.88,
        bottom=0.10, top=0.98,
        wspace=0.0, hspace=0.15
    )

    axes = [
        plt.subplot2grid((3, 1), (i, 0))
        for i in range(3)
    ]

    # shared minor ticks
    for ax in axes:
        ax.tick_params(labelsize=FONTSIZE)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which='both', direction='in',
                       bottom=True, top=True,
                       left=True, right=True)

    # x‑axis ticks at 0, π, 2π
    xt = [0, np.pi, 2*np.pi]
    xl = [r'$0$', r'$\pi$', r'$2\pi$']
    for ax in axes:
        ax.set_yticks(xt)
        ax.set_yticklabels(xl)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])

    labels = ['(a)', '(b)', '(c)']
    for lbl, ax in zip(labels, axes):
        ax.text(0.01, 0.95, lbl, transform=ax.transAxes,
                fontsize=FONTSIZE, fontweight='bold', va='top')

    extent = [0, psi.shape[1] - 1, x.min(), x.max()]

    # Panel (a): psi
    im0 = axes[0].imshow(
        psi, aspect='auto', origin='lower',
        cmap=COLORMAP, extent=extent
    )
    c0 = plt.colorbar(
        im0, ax=axes[0],
        fraction=COLORBAR_WIDTH,
        pad=COLORBAR_PAD
    )
    c0.ax.set_ylabel(r'$\psi$', fontsize=FONTSIZE, labelpad=5)
    c0.ax.tick_params(direction='in', labelsize=FONTSIZE)
    axes[0].set_ylabel(r'$x$', fontsize=FONTSIZE, labelpad=-5)

    # Panel (b): phi
    im1 = axes[1].imshow(
        phi, aspect='auto', origin='lower',
        cmap=COLORMAP, extent=extent
    )
    c1 = plt.colorbar(
        im1, ax=axes[1],
        fraction=COLORBAR_WIDTH,
        pad=COLORBAR_PAD
    )
    c1.ax.set_ylabel(r'$\phi$', fontsize=FONTSIZE, labelpad=5)
    c1.ax.tick_params(direction='in', labelsize=FONTSIZE)
    axes[1].set_ylabel(r'$x$', fontsize=FONTSIZE, labelpad=-5)

    # Panel (c): velocity
    im2 = axes[2].imshow(
        vel, aspect='auto', origin='lower',
        cmap=COLORMAP, extent=extent
    )
    c2 = plt.colorbar(
        im2, ax=axes[2],
        fraction=COLORBAR_WIDTH,
        pad=COLORBAR_PAD
    )
    c2.ax.set_ylabel(r'$v$', fontsize=FONTSIZE, labelpad=-15)
    c2.ax.tick_params(direction='in', labelsize=FONTSIZE)
    axes[2].set_xlabel(r'$t$', fontsize=FONTSIZE, labelpad=-5)
    axes[2].set_ylabel(r'$x$', fontsize=FONTSIZE, labelpad=-5)

    plt.show()
    fig.savefig(FIG_NAME, dpi=600)


def main():
    x, psi, phi, vel = load_time_series(
        DATA_FOLDER, FILE_PREFIX, NUM_FILES, FILE_EXT
    )
    plot_kymographs(x, psi, phi, vel)


if __name__ == '__main__':
    main()
