"""Diagnostic: radial pressure profile P(r) of a static droplet.

For the CSF model this shows where the pressure jump is established across
the interface band (width ~2h) and whether a clean bulk core exists at the
measurement radius - the R-dependent band/core overlap that biases the
core-annulus dP measurement at small R.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    make_couette_droplet_state,
    run,
    step,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=3.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--r-rep", type=float, default=0.45)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--mu", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--sample-from", type=int, default=400)
    args = ap.parse_args()
    R = args.R
    W = max(36.0, 2.0 * (R + 7.0))
    H = max(22.0, 2.0 * (R + 5.0))
    domain = (0.0, 0.0, W, H)        # annulus/walls far from the droplet
    n_steps = args.steps
    sample_from = args.sample_from
    every = 25
    p = SPHParams(sigma_surf=args.sigma, A_surf=10.0, B_surf=0.0,
                  r_rep=args.r_rep, switch_delta=args.delta,
                  mu_droplet=args.mu)
    s = make_couette_droplet_state(p, domain=domain, spacing=0.5,
                                   droplet_radius=R, n_wall_layers=2)
    run(s, p, 0, 0.008)
    print(f"N={s.n}  R={R}  sigma={args.sigma}  r_rep={args.r_rep}  "
          f"domain=({domain[0]},{domain[1]},{domain[2]:.0f},{domain[3]:.0f})",
          flush=True)

    # radial profile accumulators (0.25-unit bins from droplet center)
    nb = int(R * 4) + 8
    p_acc = np.zeros(nb)
    rho_acc = np.zeros(nb)
    cnt = np.zeros(nb)
    for t in range(n_steps):
        step(s, p, 0.008)
        if t >= sample_from and t % every == 0:
            in_d = s.phase == 1
            com = s.pos[in_d].mean(axis=0)
            r = np.linalg.norm(s.pos - com, axis=1)
            bins = (r * 4).astype(int)
            valid = bins < nb
            np.add.at(p_acc, bins[valid], s.pressure[valid])
            np.add.at(rho_acc, bins[valid], s.rho[valid])
            np.add.at(cnt, bins[valid], 1.0)
    mask = cnt > 0
    rb = (np.arange(nb) + 0.5) / 4.0
    idxs = np.where(mask)[0]
    pm = p_acc[idxs] / cnt[idxs]
    rm = rho_acc[idxs] / cnt[idxs]
    print("  r       P(r)     rho(r)")
    for k, i in enumerate(idxs):
        print(f"  {rb[i]:5.2f}  {pm[k]:7.4f}  {rm[k]:6.4f}")
    # center pressure (r < 1) and c~=1 interior window (r < R - 2.5h)
    c = (rb < 1.0) & mask
    interior = (rb < R - 2.5) & mask
    print(f"  center P (r<1):       {pm[c].mean():.4f}")
    print(f"  c~=1 interior P (r<{R - 2.5:.1f}): "
          f"{pm[interior].mean():.4f}  (expect p_out + sigma/R)")
    print(f"  max P at r={rb[idxs[np.argmax(pm)]]:.2f} "
          f"(P={pm.max():.4f})")


if __name__ == "__main__":
    main()
