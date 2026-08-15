"""Debug: Shepard density correction on a free hexagonal lattice."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (
    SPHParams,
    SPHState,
    build_pairs,
    compute_density,
    cubic_spline,
    hexagonal_pack,
    particle_mass,
)


def analyze(shepard: float):
    spacing = 0.5
    m = particle_mass(spacing, 1.0)
    pts = hexagonal_pack(0.0, 0.0, 14.0, 14.0, spacing)
    p = SPHParams(shepard=shepard)
    st = SPHState(pos=pts, vel=np.zeros_like(pts), mass=np.full(len(pts), m),
                  phase=np.zeros(len(pts), dtype=np.int8))
    # manual density init to control the renormalization
    pairs, d, e = build_pairs(st.pos, p.h)
    compute_density(st, p, pairs, d)
    scale = 1.0 / max(float(st.rho.mean()), 1e-9)
    st.mass *= scale
    compute_density(st, p, pairs, d)
    print(f"shepard={shepard}: mean={st.rho.mean():.4f} std={st.rho.std():.5f} "
          f"min={st.rho.min():.4f} max={st.rho.max():.4f}")
    return st.rho


r0 = analyze(0.0)
r1 = analyze(1.0)
print(f"std ratio corr/raw = {r1.std() / r0.std():.3f}")
