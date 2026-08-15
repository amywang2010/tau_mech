"""Measure the CSF force field on an equilibrated droplet, radially resolved.

For each radial bin from the droplet COM, reports the mean radial component
of the CSF acceleration (sign convention: negative = inward), its magnitude,
the color gradient |grad(c)|, and the curvature kappa = -div(n_hat). This
settles the direction/magnitude question empirically (scripts/diag_csf_sign.py
showed an expanding droplet; we need to know WHY the surface force is not
contracting it).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    build_pairs,
    compute_density,
    compute_surface_force,
    make_couette_droplet_state,
    run,
    step,
)


def main() -> None:
    p = SPHParams()
    R = 3.0
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 36.0, 22.0),
                                   spacing=0.5, droplet_radius=R)
    run(s, p, 0, 0.008)
    for _ in range(900):
        step(s, p, 0.008)

    pairs, d, e = build_pairs(s.pos, p.h)
    a_surf = compute_surface_force(s, p, pairs, d, e)

    in_d = s.phase == 1
    com = s.pos[in_d].mean(axis=0)
    rel = s.pos - com
    r = np.linalg.norm(rel, axis=1)
    rhat = rel / np.maximum(r, 1e-12)[:, None]          # outward unit vector
    a_rad = np.sum(a_surf * rhat, axis=1)               # + = outward acc

    bins = np.arange(0.0, R + 2.5, 0.5)
    idx = np.digitize(r, bins) - 1
    print(f"{'r':>5s} {'a_rad':>8s} {'|a|':>8s} {'|grad_c|':>8s} "
          f"{'kappa':>8s} {'n_parts':>7s}")
    # recompute the smoothed color gradient + kappa inline (mirror of the
    # solver: smoothed color, renormalized divergence)
    from tau_mech.sph import (cubic_spline, cubic_spline_dwdr,
                              smooth_color_field)
    c = smooth_color_field(s, p, pairs, d, n_passes=p.n_color_smooth)
    vol = s.mass / np.maximum(s.rho, p.rho_floor * s.rho0)
    dw = cubic_spline_dwdr(d, p.h)
    w = cubic_spline(d, p.h)
    w0 = cubic_spline(np.zeros(1), p.h)[0]
    i, j = pairs[:, 0], pairs[:, 1]
    # corrected gradient accumulation (same-sign, own-volume weights)
    dc_i = (c[j] - c[i]) * vol[j] * dw
    dc_j = (c[j] - c[i]) * vol[i] * dw
    n = s.n
    gx = np.bincount(np.concatenate([i, j]),
                     weights=np.concatenate([dc_i * e[:, 0], dc_j * e[:, 0]]),
                     minlength=n)
    gy = np.bincount(np.concatenate([i, j]),
                     weights=np.concatenate([dc_i * e[:, 1], dc_j * e[:, 1]]),
                     minlength=n)
    grad = np.stack([gx, gy], axis=1)
    ng = np.linalg.norm(grad, axis=1)
    nhat = grad / np.maximum(ng, 1e-12)[:, None]
    pair_term = vol[j] * np.sum((nhat[j] - nhat[i]) * (dw[:, None] * e), axis=1)
    div_n_raw = np.bincount(i, weights=pair_term, minlength=n)
    den = np.bincount(i, weights=vol[j] * w, minlength=n) + vol * w0
    div_n = div_n_raw / np.maximum(den, 1e-9)
    kappa = -div_n

    for k in range(len(bins) - 1):
        m = (idx == k) & in_d
        if m.sum() == 0:
            continue
        print(f"{0.5*(bins[k]+bins[k+1]):5.2f} "
              f"{a_rad[m].mean():+8.4f} {np.linalg.norm(a_surf[m], axis=1).mean():8.4f} "
              f"{ng[m].mean():8.4f} {kappa[m].mean():+8.4f} {int(m.sum()):7d}")
    rim = in_d & (r > R - 1.0) & (r < R + 0.2)
    print(f"\nRIM (droplet phase, {int(rim.sum())} parts):")
    print(f"  mean a_rad = {a_rad[rim].mean():+.4f}   (negative = inward)")
    print(f"  mean |a|   = {np.linalg.norm(a_surf[rim], axis=1).mean():.4f}")
    print(f"  mean kappa = {kappa[rim].mean():+.4f}   (expect ~ +1/R = +{1.0/R:.3f})")
    solv_rim = (~in_d) & (r > R) & (r < R + 1.2) & (s.phase != 2)
    print(f"  solvent-rim kappa = {kappa[solv_rim].mean():+.4f}  "
          f"a_rad = {a_rad[solv_rim].mean():+.4f}")
    print(f"  sigma/R = {p.sigma_surf / R:.4f}   "
          f"expected |a_s| ~ sigma/R/rho ~ {p.sigma_surf / R:.4f}")


if __name__ == "__main__":
    main()
