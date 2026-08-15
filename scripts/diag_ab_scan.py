"""Scan (A_surf, B_surf): stable (non-clumping) droplet with measurable dP.

Discovers the clumping instability: with B > A the smooth-switch overlap
region (0.5h-0.7h) has a net-attractive pair force, so particles collapse
into oscillating clumps (NN << spacing). Requires A >= B. Also measures the
core/annulus pressure jump to pick a strength with a resolvable Laplace
signal.
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


def probe(A: float, B: float, R: float = 3.0, n_eq: int = 1200) -> dict:
    p = SPHParams(A_surf=A, B_surf=B, mu_droplet=5.0)
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 36.0, 22.0),
                                   spacing=0.5, droplet_radius=R)
    run(s, p, 0, 0.008)
    pin_acc, pout_acc = [], []
    for t in range(n_eq):
        step(s, p, 0.008)
        if t >= 400 and t % 40 == 0:
            in_d = s.phase == 1
            com = s.pos[in_d].mean(axis=0)
            r = np.linalg.norm(s.pos - com, axis=1)
            core = in_d & (r < R - 1.5)
            far = (s.phase == 0) & (r > R + 2.0) & (r < R + 4.0)
            if core.sum() > 0 and far.sum() > 0:
                pin_acc.append(float(s.pressure[core].mean()))
                pout_acc.append(float(s.pressure[far].mean()))
    # clumping check
    pos = s.pos[s.phase == 1]
    pairs, d, e = build_pairs(pos, p.h)
    i, j = pairs[:, 0], pairs[:, 1]
    nn = np.full(len(pos), np.inf)
    np.minimum.at(nn, i, d)
    np.minimum.at(nn, j, d)
    nan = int(np.isnan(s.pos).sum())
    pin = float(np.mean(pin_acc)) if pin_acc else float("nan")
    pout = float(np.mean(pout_acc)) if pout_acc else float("nan")
    return {"A": A, "B": B, "pin": pin, "pout": pout, "dP": pin - pout,
            "nn_median": float(np.median(nn)),
            "frac_clumped": float((nn < 0.35).mean()),
            "nan": nan, "n_drop": int(len(pos))}


if __name__ == "__main__":
    for A, B in [(20.0, 20.0), (25.0, 20.0), (15.0, 10.0), (30.0, 15.0)]:
        r = probe(A, B)
        print(f"A={A:5.1f} B={B:5.1f}: dP={r['dP']:7.4f} "
              f"(pin={r['pin']:.4f} pout={r['pout']:.4f}) "
              f"clump_frac={r['frac_clumped'] * 100:5.1f}% "
              f"nn_med={r['nn_median']:.3f} nan={r['nan']}", flush=True)
