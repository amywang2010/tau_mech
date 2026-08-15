"""Direct measurement of the delivered SPH viscosity (zero free parameters).

Sets up a single-phase fluid with a sinusoidal velocity field
u(y) = U sin(k y) and evaluates the viscous acceleration a_x(y) from
compute_acceleration with pressure, surface tension and artificial viscosity
all disabled. The analytic result is

    a_x(y) = nu_eff * d^2 u/dy^2 = -nu_eff * k^2 * u(y)

so fitting the ratio a_x(y)/u(y) gives nu_eff directly. This is compared to
the nominal kinematic viscosity nu = mu_solvent / rho0. If the Morris
viscosity term carries a factor error (e.g. (mu_i+mu_j)/2 instead of
(mu_i+mu_j)), nu_eff will be off by exactly that factor.

The field is one full wavelength across the y-extent so the sinusoidal
profile is "periodic" in y (no wall truncation at the measurement band);
only the interior band (|y - Ly/2| < Ly/4) is used so the kernel is fully
supported.

Usage:
    python scripts/diag_viscosity.py [--mu 1.0] [--k 1.0]
"""
from __future__ import annotations
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tau_mech.sph import (SPHParams, SPHState, hexagonal_pack, particle_mass,
                          build_pairs, compute_density, compute_acceleration)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--spacing", type=float, default=0.25)
    args = p.parse_args()

    Ly = 2.0 * np.pi / args.k
    Lx = 2.0
    spacing = args.spacing
    params = SPHParams(mu_solvent=args.mu, mu_droplet=args.mu,
                       sigma_surf=0.0, A_surf=0.0, B_surf=0.0,
                       alpha_art=0.0, xsph=0.0, shepard=1.0)
    m = particle_mass(spacing, 1.0)
    pts = hexagonal_pack(0.0, 0.0, Lx, Ly, spacing)
    pos = pts
    phase = np.zeros(len(pos), dtype=np.int8)
    state = SPHState(pos=pos, vel=np.zeros_like(pos),
                     mass=np.full(len(pos), m), phase=phase,
                     h=params.h, rho0=1.0, domain=(0.0, 0.0, Lx, Ly))
    # initialize density + pressure (EOS at rho0 -> p ~ 0)
    pairs, d, e = build_pairs(state.pos, params.h)
    compute_density(state, params, pairs, d)
    # renormalize masses so the bulk density sits at rho0 (p -> 0)
    free = state.phase != 2
    scale = state.rho0 / max(float(state.rho[free].mean()), 1e-9)
    state.mass *= scale
    compute_density(state, params, pairs, d)

    y = state.pos[:, 1]
    y0 = 0.0
    u = 0.1 * np.sin(args.k * (y - y0))
    state.vel[:, 0] = u

    acc = compute_acceleration(state, params, pairs, d, e)
    ax = acc[:, 0]

    # interior band: kernel fully supported, away from y-edges
    band = (np.abs(y - Ly / 2.0) < Ly / 4.0)
    yy, uu, aa = y[band], u[band], ax[band]
    # a_x / u = -nu_eff k^2 ; regress a_x against -k^2 u
    # linear fit ax = slope * u  (slope = -nu_eff k^2)
    slope, *_ = np.polyfit(uu, aa, 1)
    nu_eff = -slope / args.k ** 2
    nu_nominal = args.mu / 1.0

    print(f"mu_nominal={args.mu}  rho0=1.0  nu_nominal={nu_nominal:.6f}")
    print(f"k={args.k}  Ly={Ly:.4f}  spacing={spacing}  N={len(pos)}")
    print(f"fit a_x vs u: slope={slope:.6f}  (expected -nu*k^2 = "
          f"{-nu_nominal*args.k**2:.6f})")
    print(f"nu_eff = {nu_eff:.6f}")
    print(f"ratio nu_eff/nu_nominal = {nu_eff/nu_nominal:.6f}")


if __name__ == "__main__":
    main()
