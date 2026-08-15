"""Empirical sweep: which XSPH settings keep the droplet-in-Couette state stable.

Tests xsph coefficient values and a floored-density variant of wnorm, over
300 steps; also a 2000-step no-XSPH run to check long-run stability.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from tau_mech.sph import SPHParams, make_couette_droplet_state, run


def vmax(state):
    free = state.phase != 2
    return float(np.linalg.norm(state.vel[free], axis=1).max())


def rho_stats(state):
    free = state.phase != 2
    r = state.rho[free]
    return float(r.mean()), float(r.min())


def trial(label, mutate, steps=300, dt=0.008):
    p = SPHParams()
    mutate(p)
    s = make_couette_droplet_state(p, domain=(0, 0, 24, 16), spacing=0.5,
                                   droplet_radius=2.0)
    run(s, p, 0, dt)
    run(s, p, steps, dt)
    rm, rmin = rho_stats(s)
    nan = int(np.isnan(s.pos).sum()) + int(np.isnan(s.vel).sum())
    print(f"{label:42s} steps={steps:5d}  rho_mean={rm:.3f} rho_min={rmin:.3f} "
          f"vmax={vmax(s):8.4f}  nan={nan}")
    return s


if __name__ == "__main__":
    print("=== XSPH stability sweep (droplet-in-Couette, dt=0.008) ===")
    for x in (0.5, 0.2, 0.1, 0.05, 0.02, 0.0):
        trial(f"xsph={x}", lambda p, x=x: setattr(p, "xsph", x))
    # no XSPH, long run
    trial("no XSPH, 2000 steps", lambda p: setattr(p, "xsph", 0.0), steps=2000)
    # no XSPH but everything else on, 6000 steps (Couette-like)
    trial("no XSPH, 6000 steps", lambda p: setattr(p, "xsph", 0.0), steps=6000)
