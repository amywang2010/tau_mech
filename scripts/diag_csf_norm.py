"""Pin down the CSF normal field and curvature signs on an equilibrated droplet.

Measures, per radial bin and separately for droplet-phase and solvent-phase
particles near the interface:
  nhat_r   = n_hat . r_hat   (normal radial component; -1 = inward)
  grad_r   = grad(c).r_hat / |grad(c)|   (same info from the raw gradient)
  kappa    = -div(n_hat)     (as computed by the solver's stencils)
to settle whether the discrete normal field points inward (as the smoothed
color field implies) and why kappa comes out negative on the droplet rim.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    build_pairs,
    cubic_spline,
    cubic_spline_dwdr,
    make_couette_droplet_state,
    run,
    smooth_color_field,
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
    i, j = pairs[:, 0], pairs[:, 1]
    n = s.n
    c = smooth_color_field(s, p, pairs, d, n_passes=p.n_color_smooth)
    vol = s.mass / np.maximum(s.rho, p.rho_floor * s.rho0)
    dw = cubic_spline_dwdr(d, p.h)
    w = cubic_spline(d, p.h)
    w0 = cubic_spline(np.zeros(1), p.h)[0]

    dc_i = (c[j] - c[i]) * vol[j] * dw
    dc_j = (c[j] - c[i]) * vol[i] * dw
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
    div_raw = np.bincount(i, weights=pair_term, minlength=n)
    den = np.bincount(i, weights=vol[j] * w, minlength=n) + vol * w0
    div_n = div_raw / np.maximum(den, 1e-9)
    kappa = -div_n

    in_d = s.phase == 1
    com = s.pos[in_d].mean(axis=0)
    rel = s.pos - com
    r = np.linalg.norm(rel, axis=1)
    rhat = rel / np.maximum(r, 1e-12)[:, None]
    nhat_r = np.sum(nhat * rhat, axis=1)
    grad_r = np.sum(grad * rhat, axis=1) / np.maximum(ng, 1e-12)

    for phase_name, ph in (("droplet", 1), ("solvent", 0)):
        sel = (s.phase == ph) & (r > R - 2.0) & (r < R + 2.0)
        if sel.sum() == 0:
            continue
        print(f"--- {phase_name} (n={int(sel.sum())}) ---")
        bins = np.arange(R - 2.0, R + 2.0 + 1e-9, 0.5)
        idx = np.digitize(r[sel], bins) - 1
        for k in range(len(bins) - 1):
            m = idx == k
            if m.sum() == 0:
                continue
            rb = 0.5 * (bins[k] + bins[k + 1])
            print(f"  r={rb:4.2f}: nhat_r={nhat_r[sel][m].mean():+.4f}  "
                  f"grad_r/|g|={grad_r[sel][m].mean():+.4f}  "
                  f"|g|={ng[sel][m].mean():.4f}  "
                  f"kappa={kappa[sel][m].mean():+.4f}  n={int(m.sum())}")
    print(f"\nanalytic kappa for r in [{R-2},{R+2}] should be +1/R ~ +0.33..0.5")


if __name__ == "__main__":
    main()
