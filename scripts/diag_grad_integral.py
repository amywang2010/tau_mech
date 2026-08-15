"""Direct measurement of the CSF color-gradient delta integral.

The CSF surface force is F_s = sigma * kappa * grad(c). For the Laplace law
dP = sigma/R to hold, grad(c) must act as a surface delta with radial integral

    I = integral of grad(c).r_hat across the interface = c_inside - c_outside = 1.

This script measures I directly on a STATIC droplet (no stepping) by summing
the radial component of grad(c) over the interface band, and also checks the
smoothed color field end-points (center vs far solvent). It distinguishes

  (a) a 2x under-integration of the delta function (I ~ 0.5), which would
      explain the measured sigma_eff ~ 0.46-0.52 sigma_input, from
  (b) I ~ 1.0 (delta correct; the 0.46 must come from elsewhere).

Two SPH gradient forms are compared:
  * asym:  grad_i c = sum_j V_j (c_j - c_i) grad_i W_ij   (what the code uses)
  * sym:   grad_i c = sum_j V_j (c_j + c_i) grad_i W_ij   (conservative form;
           integrates a step to ~2x the delta for the same color field)
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (
    SPHParams,
    build_pairs,
    compute_density,
    cubic_spline_dwdr,
    make_couette_droplet_state,
    smooth_color_field,
)


def main() -> None:
    p = SPHParams(sigma_surf=1.0, mu_droplet=0.5, mu_solvent=0.05)
    R = 6.0
    W = 2.0 * (R + 4.0) + 4.0
    H = 2.0 * (R + 4.0) + 4.0
    domain = (0.0, 0.0, W, H)
    state = make_couette_droplet_state(p, domain=domain, spacing=0.5,
                                       droplet_radius=R, n_wall_layers=2)
    pairs, d, e = build_pairs(state.pos, p.h, x_period=W)
    compute_density(state, p, pairs, d)

    n = state.n
    i, j = pairs[:, 0], pairs[:, 1]
    c = smooth_color_field(state, p, pairs, d, n_passes=p.n_color_smooth)
    vol = state.mass / np.maximum(state.rho, p.rho_floor * state.rho0)
    dw = cubic_spline_dwdr(d, p.h)

    # gradient (asymmetric form used by compute_surface_force)
    dc_i = (c[j] - c[i]) * vol[j] * dw
    dc_j = (c[j] - c[i]) * vol[i] * dw
    idx = np.concatenate([i, j])
    gx = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 0], dc_j * e[:, 0]]), minlength=n)
    gy = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 1], dc_j * e[:, 1]]), minlength=n)
    grad = np.stack([gx, gy], axis=1)

    # symmetric (conservative) form for comparison
    sc_i = (c[j] + c[i]) * vol[j] * dw
    sc_j = (c[j] + c[i]) * vol[i] * dw
    sx = np.bincount(idx, weights=np.concatenate([sc_i * e[:, 0], sc_j * e[:, 0]]), minlength=n)
    sy = np.bincount(idx, weights=np.concatenate([sc_i * e[:, 1], sc_j * e[:, 1]]), minlength=n)
    gsym = np.stack([sx, sy], axis=1)

    dc = state.pos - np.array([W / 2, H / 2])
    rr = np.linalg.norm(dc, axis=1)
    rhat = dc / np.maximum(rr, 1e-12)[:, None]
    free = state.phase != 2

    # radial component of gradient
    grad_r = (grad * rhat).sum(axis=1)
    gsym_r = (gsym * rhat).sum(axis=1)

    # radial bins; integrate grad_r over the interface band
    rmax = rr[free].max()
    nb = 80
    edges = np.linspace(0.0, rmax, nb + 1)
    binidx = np.digitize(rr, edges) - 1
    # area-weighted sum approximates the radial integral I = int grad_r dr
    # (in 2D, int over area of delta ~ int dr * delta * 2 pi r; here we sum the
    #  radial gradient with particle volume weighting)
    I_asym = 0.0
    I_sym = 0.0
    for k in range(nb):
        m = (binidx == k) & free
        if m.any():
            dr = edges[k + 1] - edges[k]
            I_asym += float(grad_r[m].mean()) * dr
            I_sym += float(gsym_r[m].mean()) * dr

    print("color field endpoints: c(center)=%.4f  c(far)=%.4f" %
          (float(c[rr < 1.0].mean()), float(c[rr > rmax - 1.0].mean())))
    print("grad(c) radial integral (asym, code): I = %.4f" % I_asym)
    print("grad(c) radial integral (sym, conservative): I = %.4f" % I_sym)
    print("(analytic: I = 1.0 for a color step 0 -> 1)")

    # also report the peak |grad c| and its width
    peak = float(np.abs(grad_r).max())
    print("peak |grad_r| = %.4f" % peak)


if __name__ == "__main__":
    main()
