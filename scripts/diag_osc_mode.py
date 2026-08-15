"""Characterize the droplet shape oscillation: is it a physical capillary
(Rayleigh) oscillation or a numerical artifact?

A physical quadrupole capillary oscillation of a 2D droplet has frequency
omega ~ sqrt(sigma n (n^2-1) / (rho R^3)) and is damped by viscosity with
rate ~ mu/(rho R^2). So:
  * the PERIOD must scale as 1/sqrt(sigma)
  * the DECAY must scale with mu (and it must actually decay)

A numerical ringing (e.g. a feedback in the CSF color-field smoothing, or a
discretization mode) will have a period independent of both sigma and mu.

Runs a short trajectory for a grid of (sigma, mu_droplet), records D(t), and
reports the first-overshoot time and the min/max D envelope so the period
and damping can be compared across parameters.

Usage:
    python scripts/diag_osc_mode.py --sigmas 0.5,1.0,2.0 --mus 10 --steps 4000
"""
from __future__ import annotations
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tau_mech.sph import (SPHParams, make_couette_droplet_state, run,
                          droplet_deformation, step)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", type=str, default="0.5,1.0,2.0")
    p.add_argument("--mus", type=str, default="10")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dt", type=float, default=0.008)
    p.add_argument("--every", type=int, default=100)
    args = p.parse_args()

    sigmas = [float(x) for x in args.sigmas.split(",")]
    mus = [float(x) for x in args.mus.split(",")]

    print(f"{'sigma':>6} {'mu':>5} {'D0':>7} {'D_min':>7} {'D_max':>7} "
          f"{'t_min':>7} {'t_max':>7} {'span':>7}")
    for sigma in sigmas:
        for mu in mus:
            params = SPHParams(mu_solvent=1.0, mu_droplet=mu)
            params.sigma_surf = sigma
            state = make_couette_droplet_state(
                params, domain=(0.0, 0.0, 24.0, 16.0), spacing=0.5,
                droplet_radius=3.0, n_wall_layers=4)
            run(state, params, 0, args.dt)
            Ds = []
            ts = []
            for s in range(args.steps):
                step(state, params, args.dt)
                if s % args.every == 0:
                    d = droplet_deformation(state)
                    Ds.append(d["taylor"])
                    ts.append(s * args.dt)
            Ds = np.array(Ds)
            ts = np.array(ts)
            imin = int(np.argmin(Ds))
            imax = int(np.argmax(Ds))
            print(f"{sigma:6.2f} {mu:5.1f} {Ds[0]:7.4f} {Ds[imin]:7.4f} "
                  f"{Ds[imax]:7.4f} {ts[imin]:7.1f} {ts[imax]:7.1f} "
                  f"{Ds.max()-Ds.min():7.4f}")


if __name__ == "__main__":
    main()
