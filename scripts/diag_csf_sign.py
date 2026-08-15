"""Quick CSF sign + clumping check on a single droplet (R=3).

Determines empirically whether compute_surface_force pulls the droplet
inward (Laplace dP = +sigma/R, i.e. P_core > P_solvent) or expands it
(dP < 0). Also re-checks the clumping invariant (NN-distance distribution
of the droplet phase) that the CSF model must satisfy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    build_pairs,
    make_couette_droplet_state,
    run,
    step,
)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="CSF surface tension; 0 = control without CSF")
    ap.add_argument("--A", type=float, default=10.0,
                    help="mixed-only immiscibility repulsion strength")
    ap.add_argument("--r-rep", type=float, default=0.45,
                    help="repulsion cutoff in units of h (default matches "
                         "SPHParams: short overlap barrier)")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="repulsion smooth-switch half-width in units of h")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    p = SPHParams(sigma_surf=args.sigma, A_surf=args.A, B_surf=0.0,
                  r_rep=args.r_rep, switch_delta=args.delta)
    print(f"sigma_surf={p.sigma_surf}  A_surf={p.A_surf}(mixed-only)  "
          f"r_rep={p.r_rep}  switch_delta={args.delta}  B_surf={p.B_surf}  "
          f"allow_neg_p={p.allow_neg_p}  shepard={p.shepard}")
    R = 3.0
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 36.0, 22.0),
                                   spacing=0.5, droplet_radius=R)
    run(s, p, 0, 0.008)
    pin_acc, pout_acc = [], []
    n_steps = args.steps
    for t in range(n_steps):
        step(s, p, 0.008)
        if t >= 400 and t % 25 == 0:
            in_d = s.phase == 1
            com = s.pos[in_d].mean(axis=0)
            r = np.linalg.norm(s.pos - com, axis=1)
            core = in_d & (r < R - 1.5)
            far = ((s.phase == 0) & (r > R + 2.0) & (r < R + 4.0)
                   & (s.pos[:, 1] > 2.5) & (s.pos[:, 1] < 19.5))
            if core.sum() > 0 and far.sum() > 0:
                pin_acc.append(float(s.pressure[core].mean()))
                pout_acc.append(float(s.pressure[far].mean()))
    pin, pout = float(np.mean(pin_acc)), float(np.mean(pout_acc))
    dP = pin - pout
    # clumping invariant on the droplet phase
    pos = s.pos[s.phase == 1]
    pairs, d, e = build_pairs(pos, p.h)
    nn = np.full(len(pos), np.inf)
    np.minimum.at(nn, pairs[:, 0], d)
    np.minimum.at(nn, pairs[:, 1], d)
    frac_clump = 100.0 * (nn < 0.35).mean()
    # NOTE: at R=3 the CSF transition band (~2.5-3h wide) spans the droplet,
    # so the core/annulus masks sample PARTIAL color values and the dP
    # MAGNITUDE is band-limited (~0.6-0.7x sigma/R; measured 0.166 vs 0.333).
    # This probe is a fast SIGN + stability check; the authoritative Laplace
    # verification is scripts/diag_surface_tension.py at R=5/6/7 (band-aware
    # masks, see sph._laplace_masks).
    print(f"dP = {dP:+.4f}   (sigma/R = {p.sigma_surf / R:+.4f}; R=3 is "
          f"band-limited, expect ~0.6-0.7x - authoritative check at R=5/6/7)")
    print(f"pin = {pin:.4f}   pout = {pout:.4f}   n_samples = {len(pin_acc)}")
    print(f"droplet rho = {s.rho[s.phase == 1].mean():.4f}  "
          f"solvent rho = {s.rho[s.phase == 0].mean():.4f}")
    print(f"clumping: frac(NN<0.35) = {frac_clump:.2f}%  "
          f"NN median = {np.median(nn):.4f}  n_droplet = {len(pos)}")
    print(f"nan = {int(np.isnan(s.pos).sum()) + int(np.isnan(s.vel).sum())}")
    sign_ok = dP > 0.0
    print("SIGN:", "OK (inward, dP>0)" if sign_ok
          else "WRONG (expanding, dP<0) - flip sign in compute_surface_force")
    print("CLUMP:", "OK" if frac_clump < 1.0 else "FAIL (droplet clumped)")


if __name__ == "__main__":
    main()
