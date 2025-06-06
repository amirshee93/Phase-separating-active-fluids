# Phase Separating Pattern Formation in Active Fluids

> **Phase Separating Pattern Formation in Active Fluids**  
> Amir Shee and Debasish Chaudhuri  
>  
>  
> Spontaneous phase separation and moving pattern formation in thin films of active fluids…

---

# Advection–Diffusion Simulation

This Python script numerically solves an advection–diffusion system for two species (ψ, φ) with a dynamically updating velocity field **v(x,t)**, writes time‐step snapshots to CSV, and visualizes the last step.

## Features

- Semi‑implicit time stepping (multi‑step Adams–Bashforth / Crank–Nicolson)
- Cyclic Thomas algorithm for tridiagonal solves
- Periodic boundary conditions
- Writes concentration & velocity profiles every step into `dynamics_data/`
- Summary metadata & elapsed time saved to `input_output.csv`
- Optional Matplotlib animation of the last frame

## Prerequisites

- Python 3.7+
- NumPy
- Matplotlib

# Stability Maps (fig1ab.py)

This script generates a two‑panel figure showing the real and imaginary parts of the growth rate
(λ) for an active–passive system as functions of wave number *q* and parameters *Pe* or *ωₒₙ*.

# Heatmap of $q_{\max}$ over $\omega_{\mathrm{on}}$–Pe Plane (fig1c.py)

This script computes and plots the wavenumber $q$ that maximizes the growth rate
$\lambda_{\mathrm{Re}}(q)$ as a function of on‐rate $\omega_{\rm on}$ and Peclet
number Pe, producing panel (c) of Figure 1.

# Out-of-phase oscillatory-moving concentrations & velocity (fig2.py)

This script loads a time series of CSV files containing  
`x, ψ(x,t), φ(x,t), v(x,t)` data and produces three stacked  
kymograph panels (ψ, φ, and v) as `fig2.png`.

# Stationary in-phase and out-of-phase concentration & velocity profiles (fig3.py)

This script generates a 2×2 panel figure showing concentrations (ψ, φ) and velocity derivatives
for two parameter sets (Pe=3.5, ωₒₙ=0.15 in panels a/c; Pe=3.0, ωₒₙ=0.5 in panels b/d).  
The output is saved as `fig3.png`.

# Phase Diagram & Stability Map (fig4.py)

This script computes and visualizes the real‐part zero contour, oscillatory regime,  
and phase‐diagram points on the (ωₒₙ, Pe) plane, saving  
the result as `fig4.png`

