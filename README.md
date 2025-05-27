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


# Stability Maps: Fig1ab.py

This script generates a two‑panel figure showing the real and imaginary parts of the growth rate
(λ) for an active–passive system as functions of wave number *q* and parameters *Pe* or *ωₒₙ*.

# Heatmap of $q_{\max}$ over $\omega_{\mathrm{on}}$–Pe Plane

This script computes and plots the wavenumber $q$ that maximizes the growth rate
$\lambda_{\mathrm{Re}}(q)$ as a function of on‐rate $\omega_{\rm on}$ and Peclet
number Pe, producing panel (c) of Figure 1.
