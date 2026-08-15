"""Probe: Laplace dP equilibration trajectory of one CSF droplet.

Tests the hypothesis that the earlier calibration (damp_mu=5) under-measured
dP because the runs were trapped at ~0.3-0.4x the capillary relaxation time
t_char = mu*R/sigma_eff. Runs R=5 at the study viscosity (mu=0.5,
t_char ~ 5 units) and prints the dP trajectory over time windows: if the
hypothesis is right, dP rises to a plateau near sigma_eff/R (0.5*0.2 = 0.1)
well within the run.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    _laplace_masks,
    make_couette_droplet_state,
    run,
    step,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--window", type=int, default=400)
    args = ap.parse_args()
    p = SPHParams(sigma_surf=args.sigma, mu_droplet=args.mu)
    R = args.R
    W = 2.0 * (R + 4.0) + 4.0
    H = 2.0 * (R + 4.0) + 4.0
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, W, H), spacing=0.5,
                                   droplet_radius=R, n_wall_layers=2)
    run(s, p, 0, 0.008)
    t_char = args.mu * R / (0.5 * args.sigma)  # sigma_eff ~ 0.5 sigma
    print(f"R={R}  mu={args.mu}  sigma={args.sigma}  t_char~{t_char:.1f}  "
          f"t_run={args.steps * 0.008:.1f}  ({args.steps * 0.008 / t_char:.1f}x "
          f"t_char)", flush=True)
    print("  t       dP      pin     pout    n_core n_far")
    win = max(1, args.window)
    for t0 in range(0, args.steps, win):
        for _ in range(win):
            step(s, p, 0.008)
        core, far = _laplace_masks(s, R, p.h)
        if core.sum() > 0 and far.sum() > 0:
            pin = float(s.pressure[core].mean())
            pout = float(s.pressure[far].mean())
            print(f"  {(t0 + win) * 0.008:5.1f}  {pin - pout:+7.4f}  "
                  f"{pin:+7.4f}  {pout:+7.4f}  {int(core.sum()):5d}  "
                  f"{int(far.sum()):5d}", flush=True)
    print(f"  (Laplace expectation sigma/R = {args.sigma / R:.4f}; "
          f"half-band estimate 0.5*sigma/R = {0.5 * args.sigma / R:.4f})")


if __name__ == "__main__":
    main()
