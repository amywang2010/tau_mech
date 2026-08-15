"""Diagnostic: isolate which SPH force term drives the exponential
velocity instability observed in the droplet-in-Couette state.

Each config is run for N steps from the mass-renormalized initial state;
we report rho stats, max free-particle velocity, and NaNs.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from tau_mech.sph import SPHParams, make_couette_droplet_state, run


def run_config(label, mutate, steps=30, dt=0.008, print_trace=False):
    p = SPHParams()
    mutate(p)
    s = make_couette_droplet_state(p, domain=(0, 0, 24, 16), spacing=0.5,
                                   droplet_radius=2.0)
    run(s, p, 0, dt)  # density init + mass renormalization
    trace = []
    for k in range(1, steps + 1):
        run(s, p, 1, dt)
        if print_trace or k == steps:
            free = s.phase != 2
            v = np.linalg.norm(s.vel[free], axis=1)
            rho = s.rho[free]
            trace.append((k, rho.mean(), rho.min(), v.max()))
    last = trace[-1]
    nan = int(np.isnan(s.pos).sum()) + int(np.isnan(s.vel).sum())
    print(f"{label:34s} steps={steps}  rho_mean={last[1]:.3f} "
          f"rho_min={last[2]:.3f}  vmax={last[3]:.4f}  nan={nan}")
    if print_trace:
        for k, rm, rmin, vmax in trace:
            print(f"    step {k:3d}: rho_mean={rm:.3f} rho_min={rmin:.3f} vmax={vmax:.4f}")
    return last[3]


if __name__ == "__main__":
    print("=== SPH instability bisection (droplet-in-Couette, dt=0.008) ===")
    run_config("base (all terms on)", lambda p: None, print_trace=True)
    run_config("pressure ~ off (c_s=0.1)", lambda p: setattr(p, "c_s", 0.1))
    run_config("no surface tension", lambda p: (setattr(p, "A_surf", 0.0),
                                                setattr(p, "B_surf", 0.0)))
    run_config("no XSPH", lambda p: setattr(p, "xsph", 0.0))
    run_config("no art. viscosity", lambda p: setattr(p, "alpha_art", 0.0))
    run_config("pressure only", lambda p: (setattr(p, "c_s", 0.1),
                                           setattr(p, "A_surf", 0.0),
                                           setattr(p, "B_surf", 0.0),
                                           setattr(p, "mu_solvent", 0.0),
                                           setattr(p, "mu_droplet", 0.0),
                                           setattr(p, "alpha_art", 0.0),
                                           setattr(p, "xsph", 0.0)))
    run_config("surf only", lambda p: (setattr(p, "c_s", 0.1),
                                       setattr(p, "mu_solvent", 0.0),
                                       setattr(p, "mu_droplet", 0.0),
                                       setattr(p, "alpha_art", 0.0),
                                       setattr(p, "xsph", 0.0)))
    run_config("softer EOS c_s=3", lambda p: setattr(p, "c_s", 3.0))
    run_config("no droplet surf on solvent", lambda p: (setattr(p, "A_surf", 0.0),
                                                        setattr(p, "B_surf", 0.0),
                                                        setattr(p, "c_s", 3.0)))
