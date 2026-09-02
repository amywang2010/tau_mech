"""Resolution-convergence probe for the CSF curvature operator.

Distinguishes a symmetry DEFECT (label/direction dependent - fixed 2026-09-02,
see scripts/diag_csf_symmetry.py) from finite-resolution DISCRETIZATION bias
(which must converge as h/R -> 0). On a circular droplet the rim curvature
must approach kappa = 1/R with an azimuthal spread that shrinks with
resolution. If the spread did NOT shrink with h/R, the residual anisotropy
would be a model defect, not a discretization artifact.

Two resolution axes are probed independently:
  * droplet radius R at fixed h (h/R decreases)
  * lattice spacing at fixed R (via particle_mass renormalization the density
    stays at rho0, so only the discretization changes)

Run:  python scripts/diag_csf_convergence.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    build_pairs,
    color_field_curvature,
    compute_density,
    make_couette_droplet_state,
    run,
)


def rim_stats(params, radius, spacing, domain):
    state = make_couette_droplet_state(params, domain=domain, spacing=spacing,
                                       droplet_radius=radius, n_wall_layers=4)
    run(state, params, 0, 0.008)
    pairs, d, e = build_pairs(state.pos, params.h,
                              x_period=state.domain[2] - state.domain[0])
    compute_density(state, params, pairs, d)
    kappa, grad, _ = color_field_curvature(state, params, pairs, d, e)
    com = state.pos[state.phase == 1].mean(axis=0)
    rel = state.pos - com
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
    gnorm = np.linalg.norm(grad, axis=1)
    droplet = state.phase == 1
    rim = droplet & (gnorm > 0.25 * gnorm[droplet].max())
    n_sec = 12
    means, counts = [], []
    for k in range(n_sec):
        m = rim & (az >= 360.0 / n_sec * k) & (az < 360.0 / n_sec * (k + 1))
        counts.append(int(m.sum()))
        means.append(float(kappa[m].mean()) if m.sum() >= 3 else np.nan)
    means = np.array(means)
    fin = means[np.isfinite(means)]
    expected = 1.0 / radius
    return {
        "radius": radius,
        "spacing": spacing,
        "h_over_R": params.h / radius,
        "expected_kappa": expected,
        "rim_mean_kappa": float(np.mean(fin)),
        "rim_mean_over_expected": float(np.mean(fin) / expected),
        "azimuthal_spread": float(fin.max() - fin.min()),
        "spread_over_expected": float((fin.max() - fin.min()) / expected),
        "n_rim": int(rim.sum()),
        "min_sector_count": int(min(counts)),
    }


def main():
    params = SPHParams()
    rows = []
    # axis 1: R at fixed spacing (h/R: 0.333 -> 0.143)
    for R in (3.0, 4.0, 5.0, 6.0, 7.0):
        W = 2.0 * (R + 5.0) + 4.0
        rows.append(rim_stats(params, R, 0.5, (0.0, 0.0, W, W)))
    # axis 2: spacing at fixed R = 6 (0.5 -> 0.25 -> 0.125; particle count ~ x4 each step)
    for spacing in (0.5, 0.25):
        W = 2.0 * (6.0 + 5.0) + 4.0
        rows.append(rim_stats(params, 6.0, spacing, (0.0, 0.0, W, W)))
    print("=== CSF curvature resolution convergence ===")
    print(f"{'R':>5} {'spacing':>8} {'h/R':>7} {'<k>R':>7} {'<k>R/1':>8} "
          f"{'spread':>8} {'spread/(1/R)':>13} {'n_rim':>6}")
    for r in rows:
        print(f"{r['radius']:5.1f} {r['spacing']:8.3f} {r['h_over_R']:7.3f} "
              f"{r['rim_mean_kappa'] * r['radius']:7.3f} "
              f"{r['rim_mean_over_expected']:8.3f} "
              f"{r['azimuthal_spread']:8.4f} {r['spread_over_expected']:13.3f} "
              f"{r['n_rim']:6d}")
    out = {"rows": rows,
           "interpretation": "defect fixed if spread/(1/R) decreases with "
                              "resolution; residual is discretization bias"}
    os.makedirs("outputs/sph/audits", exist_ok=True)
    with open("outputs/sph/audits/csf_convergence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> outputs/sph/audits/csf_convergence.json")


if __name__ == "__main__":
    main()
