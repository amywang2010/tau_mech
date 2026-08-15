"""Probe: particle clumping / pairing instability check in the droplet.

If the TM attraction pins particles into clumps (nearest-neighbor distances
well below the lattice spacing 0.5), the "fluid" is actually a cluster
solid, which would explain the compression-shell pressure structure and the
size-dependent effective surface tension. A healthy fluid keeps the
lattice spacing.
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
)


def main() -> None:
    p = SPHParams(A_surf=10.0, B_surf=20.0, mu_droplet=5.0)
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 36.0, 22.0),
                                   spacing=0.5, droplet_radius=5.0)
    run(s, p, 0, 0.008)
    run(s, p, 4000, 0.008)  # long equilibration (32 time units)

    in_d = s.phase == 1
    pos = s.pos[in_d]
    print(f"droplet particles: {len(pos)}")
    pairs, d, e = build_pairs(pos, p.h)
    # nearest-neighbour distance per particle
    i, j = pairs[:, 0], pairs[:, 1]
    nn = np.full(len(pos), np.inf)
    np.minimum.at(nn, i, d)
    np.minimum.at(nn, j, d)
    print(f"NN distance:  min={nn.min():.4f}  p10={np.percentile(nn, 10):.4f}  "
          f"median={np.median(nn):.4f}  (lattice spacing = 0.5)")
    frac_clumped = float((nn < 0.35).mean())
    print(f"fraction with NN < 0.35 (clumped): {frac_clumped * 100:.2f}%")
    # pair correlation g(r) histogram
    hist, edges = np.histogram(d, bins=np.linspace(0, 2.0, 41))
    r_mid = (edges[:-1] + edges[1:]) / 2
    print("g(r): " + " ".join(f"{r_mid[k]:.2f}:{hist[k]}" for k in range(0, 40, 4)))


if __name__ == "__main__":
    main()
