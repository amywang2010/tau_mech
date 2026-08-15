"""Probe: how many steps does the R=3 droplet need to reach its round
equilibrium (discretization floor) before shear? Reports D(t) so the sweep's
eq_steps can be set above the transient (the zero-shear control must show a
flat D, not a decaying one)."""
from __future__ import annotations
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tau_mech.sph import (SPHParams, make_couette_droplet_state, run,
                          droplet_deformation)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--dt", type=float, default=0.008)
    p.add_argument("--mu-solvent", type=float, default=1.0)
    p.add_argument("--mu-droplet", type=float, default=10.0)
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--sigma", type=float, default=None,
                   help="surface tension (default: SPHParams.sigma_surf)")
    args = p.parse_args()

    params = SPHParams(mu_solvent=args.mu_solvent, mu_droplet=args.mu_droplet)
    if args.sigma is not None:
        params.sigma_surf = args.sigma
    state = make_couette_droplet_state(params, domain=(0.0, 0.0, 24.0, 16.0),
                                       spacing=0.5, droplet_radius=3.0,
                                       n_wall_layers=4)
    run(state, params, 0, args.dt)
    print("step  t       D       aspect")
    from tau_mech.sph import step
    for s in range(args.steps):
        step(state, params, args.dt)
        if s % args.every == 0:
            d = droplet_deformation(state)
            print(f"{s:5d} {s*args.dt:7.2f} {d['taylor']:.4f} "
                  f"{d['aspect_ratio']:.4f}")
    d = droplet_deformation(state)
    print(f"FINAL {args.steps:5d} {args.steps*args.dt:7.2f} "
          f"{d['taylor']:.4f} {d['aspect_ratio']:.4f}")


if __name__ == "__main__":
    main()
