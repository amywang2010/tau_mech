"""Direct curvature diagnostic: is kappa ~ 1/R (correct) or ~ 0.5/R?

Measures the Adami-2010 reproducing-divergence curvature at the interface of
a STATIC circular droplet (no time stepping, so it is cheap and unambiguous)
and compares it against the analytic 2D value kappa = 1/R. It also prints the
two candidate renormalization denominators:

  D_W    = sum_j V_j W_ij              (kernel-sum, ~ +1)
  D_rW   = sum_j V_j (r_ij . grad W)   (Adami 2010 reproducing denominator, ~ -2 in 2D)

If kappa ~ 0.5/R while D_W ~ 1 and D_rW ~ -2, the curvature is under-delivered
by the wrong renormalization denominator, which would explain the observed
sigma_eff ~ 0.46-0.52 sigma_input in the Laplace verification.
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
    cubic_spline,
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
    w = cubic_spline(d, p.h)
    w0 = cubic_spline(np.zeros(1), p.h)[0]

    # gradient (same as compute_surface_force)
    dc_i = (c[j] - c[i]) * vol[j] * dw
    dc_j = (c[j] - c[i]) * vol[i] * dw
    idx = np.concatenate([i, j])
    gx = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 0], dc_j * e[:, 0]]), minlength=n)
    gy = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 1], dc_j * e[:, 1]]), minlength=n)
    grad = np.stack([gx, gy], axis=1)
    ng = np.linalg.norm(grad, axis=1)
    nhat = grad / np.maximum(ng, 1e-12)[:, None]

    # divergence (same as compute_surface_force)
    pair_term = vol[j] * np.sum((nhat[j] - nhat[i]) * (dw[:, None] * e), axis=1)
    div_n_raw = np.bincount(i, weights=pair_term, minlength=n)
    D_W = np.bincount(i, weights=vol[j] * w, minlength=n) + vol * w0
    D_rW = np.bincount(i, weights=vol[j] * (d * dw), minlength=n)

    kappa_code = -div_n_raw / np.maximum(D_W, 1e-9)
    kappa_adami = -div_n_raw / np.maximum(D_rW, 1e-9)

    # interface particles: those with the largest |grad c|
    dc = np.asarray((state.pos - np.array([W / 2, H / 2])))
    rr = np.linalg.norm(dc, axis=1)
    interface = (ng > 0.05 * ng.max()) & (state.phase != 2)
    # exclude wall band
    free = state.phase != 2
    sel = interface & free
    if sel.sum() == 0:
        sel = free

    k = kappa_code[sel]
    ka = kappa_adami[sel]
    rs = rr[sel]
    print("interface particles:", int(sel.sum()))
    print("  kappa_code  mean*R = %.4f   median*R = %.4f" % (np.mean(k * rs), np.median(k * rs)))
    print("  kappa_adami mean*R = %.4f   median*R = %.4f" % (np.mean(ka * rs), np.median(ka * rs)))
    print("  (analytic: kappa*R = 1.000 for a circle)")
    # denominator at interface
    print("  D_W  (kernel-sum)      mean = %.4f" % np.mean(D_W[sel]))
    print("  D_rW (Adami r.gradW)   mean = %.4f" % np.mean(D_rW[sel]))
    # per-particle scatter for the best-resolved interface shell
    mid = (rs > R - 1.5) & (rs < R + 1.5)
    if mid.sum() > 0:
        print("  |r-R|<1.5 shell: kappa_code*R mean=%.3f std=%.3f | kappa_adami*R mean=%.3f std=%.3f"
              % (np.mean(k[mid] * rs[mid]), np.std(k[mid] * rs[mid]),
                 np.mean(ka[mid] * rs[mid]), np.std(ka[mid] * rs[mid])))


if __name__ == "__main__":
    main()
