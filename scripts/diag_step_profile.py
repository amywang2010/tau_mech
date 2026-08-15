"""Line-by-line timing of one SPH step (find the unaccounted ~400 ms)."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from tau_mech.sph import (  # noqa: E402
    SPHParams, build_pairs, compute_acceleration, compute_density,
    cubic_spline, make_couette_droplet_state, run,
)

p = SPHParams(A_surf=10.0, B_surf=20.0, mu_droplet=5.0)
s = make_couette_droplet_state(p, domain=(0, 0, 36, 22), spacing=0.5,
                               droplet_radius=3.0)
run(s, p, 0, 0.008)


def T(label, fn):
    t0 = time.time()
    r = fn()
    print(f"{label:<28} {(time.time()-t0)*1000:7.1f} ms")
    return r


pairs, d, e = T("build_pairs #1", lambda: build_pairs(s.pos, p.h))
a0 = T("acceleration #1", lambda: compute_acceleration(s, p, pairs, d, e))
s.vel = s.vel + 0.5 * 0.008 * a0
T("vel update", lambda: s.vel + 0.5 * 0.008 * a0)
s.pos = s.pos + 0.008 * s.vel
T("pos update", lambda: s.pos + 0.008 * s.vel)
pairs, d, e = T("build_pairs #2", lambda: build_pairs(s.pos, p.h))
T("density", lambda: compute_density(s, p, pairs, d))
a1 = T("acceleration #2", lambda: compute_acceleration(s, p, pairs, d, e))
s.vel = s.vel + 0.5 * 0.008 * a1

# XSPH block copied verbatim
t0 = time.time()
rho_floor = p.rho_floor * s.rho0
rho_i = np.maximum(s.rho[pairs[:, 0]], rho_floor)
rho_j = np.maximum(s.rho[pairs[:, 1]], rho_floor)
w = cubic_spline(d, p.h)
cij = p.xsph * s.mass[pairs[:, 1]] * w / (0.5 * (rho_i + rho_j))
dv = (s.vel[pairs[:, 1]] - s.vel[pairs[:, 0]]) * cij[:, None]
print(f"{'XSPH setup':<28} {(time.time()-t0)*1000:7.1f} ms")
free_i = s.phase[pairs[:, 0]] != 2
free_j = s.phase[pairs[:, 1]] != 2
t0 = time.time()
if free_i.any():
    idx = pairs[free_i, 0]
    s.vel[:, 0] += np.bincount(idx, weights=dv[free_i, 0], minlength=s.n)
    s.vel[:, 1] += np.bincount(idx, weights=dv[free_i, 1], minlength=s.n)
if free_j.any():
    idx = pairs[free_j, 1]
    s.vel[:, 0] -= np.bincount(idx, weights=dv[free_j, 0], minlength=s.n)
    s.vel[:, 1] -= np.bincount(idx, weights=dv[free_j, 1], minlength=s.n)
print(f"{'XSPH bincounts':<28} {(time.time()-t0)*1000:7.1f} ms")
