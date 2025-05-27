
---

### `active_fluids.py`
```python
#!/usr/bin/env python3
"""
Advection–Diffusion Simulation with CSV Output

Solves coupled advection–diffusion for two concentrations (ψ, φ)
with a dynamic velocity field v(x,t). Writes profiles to CSV and
prints elapsed runtime.
"""

import os
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ──────────────────────────────────────────────────────────────────────────────
# Simulation parameters
L = 2 * np.pi       # Domain length
dx = 0.01           # Grid spacing
m = int(L / dx)     # Number of grid points

al = 0.1            # Passive diffusion ratio
rs = 10.0           # Off‑rate sensitivity
wo = 1.0            # On‑rate
won = 0.5           # Off‑rate
pe = 2.6            # Péclet number

dt = 0.01           # Time step
kappa = dt / dx**2  # Diffusion number
gam = dt / (2.0 * dx)

steps       = 100    # inner loop per frame
skip_steps  = 0
max_steps   = skip_steps + 1000
# ──────────────────────────────────────────────────────────────────────────────

# Spatial grid and ICs
xx   = np.linspace(0, L, m)
cao  = np.full(m, won / (won + wo))
cpo  = np.full(m, wo  / (won + wo))
vo   = 0.0001 * np.sin(xx)

# “Old” copies for multi‑step scheme
caoo, cpoo = cao.copy(), cpo.copy()
voo        = np.zeros(m)


def thomas(a, b, c, f):
    """
    Solve cyclic tridiagonal system via a Thomas‑like algorithm.
    a, b, c : sub/main/super diagonals (length m)
    f       : RHS, returns solution in f
    """
    gam2 = np.zeros_like(a, dtype=np.float64)

    # Forward elimination
    b[0] = 1.0 / b[0]
    gam2[0] = -a[0] * b[0]
    a[0] = f[0] * b[0]

    for i in range(1, m - 2):
        cim1 = i - 1
        c[cim1] *= b[cim1]
        denom     = b[i] - a[i] * c[cim1]
        b[i]      = 1.0 / denom
        gam2[i]   = -a[i] * gam2[cim1] * b[i]
        a[i]      = (f[i] - a[i] * a[cim1]) * b[i]

    gam2[m - 3] -= c[m - 3] * b[m - 3]

    # Back substitution
    f[m - 3] = a[m - 3]
    b[m - 3] = gam2[m - 3]
    for k1 in range(1, m - 2):
        k  = m - 3 - k1
        k2 = k + 1
        f[k] = a[k] - c[k] * f[k2]
        b[k] = gam2[k] - c[k] * b[k2]

    # Last eqn
    k1, k2 = m - 2, m - 3
    zaa = (f[k1] - c[k1] * f[0] - a[k1] * f[k2]) \
        / (b[k1] + a[k1] * b[k2] + c[k1] * b[0])
    f[k1] = zaa
    for i in range(m - 2):
        f[i] += b[i] * zaa

    # Enforce periodicity
    f[m - 1] = f[0]
    return f


def mainloop(frame):
    """Perform `steps` sub‑steps, update fields, write CSV, update plot."""
    global cao, cpo, vo, caoo, cpoo, voo

    for _ in range(steps):
        # Source term
        src = np.zeros(m)
        for j in range(m):
            jp1 = j + 1 if j < m - 1 else 1
            jm1 = j - 1 if j > 0     else m - 1
            div_v = (vo[jp1] - vo[jm1]) / (2 * dx)
            woff  = wo * np.exp(rs * div_v)
            src[j] = (1.5 * (woff * cao[j] - won * cpo[j])
                      - 0.5 * (woff * caoo[j] - won * cpoo[j]))

        # Active species
        aa = np.full(m, -9/16 * kappa)
        bb = np.full(m, 1 + 9/8 * kappa)
        cc = aa.copy()
        ff = np.zeros(m)
        for j in range(m):
            jp1 = j + 1 if j < m - 1 else 1
            jm1 = j - 1 if j > 0     else m - 1
            d1  = kappa * (cao[jp1] + cao[jm1] - 2 * cao[j])
            d2  = kappa * (caoo[jp1] + caoo[jm1] - 2 * caoo[j])
            a1  = -gam * (cao[j] * (vo[jp1]-vo[jm1])
                          + vo[j] * (cao[jp1]-cao[jm1]))
            a2  = -gam * (caoo[j] * (voo[jp1]-voo[jm1])
                          + voo[j] * (caoo[jp1]-caoo[jm1]))
            ff[j] = cao[j] + 1.5*a1 - 0.5*a2 + 0.375*d1 + 0.0625*d2 - dt * src[j]
        ca = thomas(aa.copy(), bb.copy(), cc.copy(), ff.copy())

        # Passive species
        aa.fill(-al * 9/16 * kappa)
        bb.fill(1 + al * 9/8 * kappa)
        cc.fill(aa[0])
        ff.fill(0)
        for j in range(m):
            jp1 = j + 1 if j < m - 1 else 1
            jm1 = j - 1 if j > 0     else m - 1
            d1  = al * kappa * (cpo[jp1] + cpo[jm1] - 2 * cpo[j])
            d2  = al * kappa * (cpoo[jp1] + cpoo[jm1] - 2 * cpoo[j])
            a1  = -gam * (cpo[j] * (vo[jp1]-vo[jm1])
                          + vo[j] * (cpo[jp1]-cpo[jm1]))
            a2  = -gam * (cpoo[j] * (voo[jp1]-voo[jm1])
                          + voo[j] * (cpoo[jp1]-cpoo[jm1]))
            ff[j] = cpo[j] + 1.5*a1 - 0.5*a2 + 0.375*d1 + 0.0625*d2 + dt * src[j]
        cp = thomas(aa.copy(), bb.copy(), cc.copy(), ff.copy())

        # Velocity update (force balance)
        aa.fill(1)
        bb.fill(-(2 + dx*dx))
        cc.fill(1)
        ff.fill(0)
        for j in range(m):
            jp1 = j + 1 if j < m - 1 else 1
            jm1 = j - 1 if j > 0     else m - 1
            ff[j] = -(pe / (1 + cao[j])**2) * ((cao[jp1]-cao[jm1]) * (dx/2))
        v = thomas(aa.copy(), bb.copy(), cc.copy(), ff.copy())

        # Rotate fields
        caoo, cpoo, voo = cao.copy(), cpo.copy(), vo.copy()
        cao, cpo, vo    = ca,       cp,     v

    # write CSV for this frame
    if skip_steps <= frame <= max_steps:
        folder = "dynamics_data"
        os.makedirs(folder, exist_ok=True)
        fname = f"{folder}/concentrations_velocity_{frame-skip_steps}.csv"
        np.savetxt(
            fname,
            np.column_stack((xx, cao, cpo, vo)),
            delimiter=",",
            header="x,psi,phi,v",
            comments=""
        )

    # update plot
    line1.set_ydata(cao)
    line2.set_ydata(cpo)
    line3.set_ydata(vo)
    ax1.set_ylim(min(cao.min(), cpo.min())*1.1,
                 max(cao.max(), cpo.max())*1.1)
    ax2.set_ylim(vo.min()*1.1, vo.max()*1.1)
    return line1, line2, line3


def main():
    start = time.time()

    # CSV metadata header
    with open("input_output.csv", "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["alpha", al])
        writer.writerow(["won",   won])
        writer.writerow(["rs",    rs])
        writer.writerow(["wo",    wo])
        writer.writerow(["dt",    dt])
        writer.writerow(["jump",  steps])
        writer.writerow(["skip",  skip_steps])
        writer.writerow(["total", max_steps])

    # Plot setup
    global fig, ax1, ax2, line1, line2, line3
    fig, ax1 = plt.subplots()
    ax2     = ax1.twinx()
    line1, = ax1.plot(xx, cao, "b", label="ψ(x,t)")
    line2, = ax1.plot(xx, cpo, "g", label="φ(x,t)")
    line3, = ax2.plot(xx, vo,  "r--", label="v(x,t)")

    ax1.set_xlabel("Position")
    ax1.set_ylabel("Concentration")
    ax2.set_ylabel("Velocity")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Advection–Diffusion with Dynamic Velocity")

    # Uncomment to animate last frame
    # ani = FuncAnimation(fig, mainloop, frames=max_steps, blit=False)
    # plt.show()

    # run and write data
    for t in range(max_steps):
        mainloop(t)

    elapsed = time.time() - start
    # append elapsed time
    with open("input_output.csv", "a", newline="") as out:
        csv.writer(out).writerow(["elapsed_sec", f"{elapsed:.2f}"])

    print(f"Done! Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
