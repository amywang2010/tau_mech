"""Measure the CSF surface-force integral, which sets the Laplace jump.

At equilibrium grad P = F_s = sigma * kappa * grad(c), so the pressure jump is

    dP = P_in - P_out = - sigma * integral of (kappa * grad_r c) dr

across the interface (grad_r = grad(c).r_hat, negative because grad(c) points
inward). For a circle kappa = 1/R and the delta integrates to 1, so
dP = sigma/R. This script computes the radial integral of kappa * grad_r c
directly on a STATIC droplet and reports the implied sigma_eff = dP*R, plus
the force-weighted mean curvature <kappa> = int(kappa |grad c|)/int(|grad c|).
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

    dc_i = (c[j] - c[i]) * vol[j] * dw
    dc_j = (c[j] - c[i]) * vol[i] * dw
    idx = np.concatenate([i, j])
    gx = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 0], dc_j * e[:, 0]]), minlength=n)
    gy = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 1], dc_j * e[:, 1]]), minlength=n)
    grad = np.stack([gx, gy], axis=1)
    ng = np.linalg.norm(grad, axis=1)
    nhat = grad / np.maximum(ng, 1e-12)[:, None]

    pair_term = vol[j] * np.sum((nhat[j] - nhat[i]) * (dw[:, None] * e), axis=1)
    div_n_raw = np.bincount(i, weights=pair_term, minlength=n)
    den = np.bincount(i, weights=vol[j] * w, minlength=n) + vol * w0
    kappa = -div_n_raw / np.maximum(den, 1e-9)

    dc = state.pos - np.array([W / 2, H / 2])
    rr = np.linalg.norm(dc, axis=1)
    rhat = dc / np.maximum(rr, 1e-12)[:, None]
    grad_r = (grad * rhat).sum(axis=1)
    free = state.phase != 2

    # force-weighted mean curvature at the interface (weight by |grad c|)
    wgt = np.abs(grad_r) * free
    wsum = wgt.sum()
    kappa_wmean = float((kappa * wgt).sum() / max(wsum, 1e-12))
    print("|grad c|-weighted mean kappa * R = %.4f" % (kappa_wmean * R))
    print("(analytic 1.0)")

    # radial integral of kappa * grad_r c  ->  -dP/sigma
    rmax = rr[free].max()
    nb = 120
    edges = np.linspace(0.0, rmax, nb + 1)
    binidx = np.digitize(rr, edges) - 1
    I = 0.0
    for k in range(nb):
        m = (binidx == k) & free
        if m.any():
            dr = edges[k + 1] - edges[k]
            I += float((kappa[m] * grad_r[m]).mean()) * dr
    dP_implied = -p.sigma_surf * I
    print("integral of kappa*grad_r dr = %.4f  (=> dP/sigma = %.4f, sigma_eff=%.4f)"
          % (I, -I, dP_implied * R))
    print("(analytic: integral = -1/R = %.4f, sigma_eff = 1.0)" % (-1.0 / R))


if __name__ == "__main__":
    main()
