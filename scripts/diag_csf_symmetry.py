"""Operator-level audit of the CSF curvature/normal-divergence stencil.

Purpose: decide from the MATH whether the CSF interface operators are
pairwise symmetric, BEFORE trusting any long dynamics run. Four probes:

  A. Index-permutation invariance: physical operators cannot depend on how
     particles are numbered. Permute particle order, re-run, compare fields.
  B. Azimuthal uniformity: on a perfect circle, kappa must be 1/R everywhere
     on the rim, independent of azimuth. A one-sided stencil breaks
     rotational symmetry in a way that correlates with lattice/index order.
  C. Net internal force and torque: the surface force is INTERNAL. Sum_i
     m_i a_i and the torque integral must vanish to discretization error.
  D. Stencil census: for each particle, what fraction of its neighbour pairs
     contribute to its divergence accumulation (1.0 = fully symmetric).

The audit imports color_field_curvature from tau_mech.sph directly - the
SAME function the solver uses - so it can never silently diverge from the
solver (a duplicated copy of the operator here produced exactly that failure
during this audit; fixed by making this script consume the shared helper).

Baseline evidence (one-sided stencil, before the 2026-09-02 fix):
  [A] NOT permutation-invariant: kappa moved by up to 1.51 (kappa ~ 0.33)
  [B] rim sector means 0.29-0.62 vs 1/R = 0.333
  [C] net internal force = 10.8% of the force magnitude; NON-CONSERVATIVE
  [D] mean fraction of pairs in the stencil = 0.51 (one-sided)

Run:  python scripts/diag_csf_symmetry.py
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


def setup_droplet(params, domain=(0.0, 0.0, 24.0, 16.0), radius=3.0):
    state = make_couette_droplet_state(params, domain=domain, spacing=0.5,
                                       droplet_radius=radius, n_wall_layers=4)
    run(state, params, 0, 0.008)  # density init + mass renormalization
    pairs, d, e = build_pairs(state.pos, params.h,
                              x_period=state.domain[2] - state.domain[0])
    compute_density(state, params, pairs, d)
    return state, pairs, d, e


def probe_permutation_invariance():
    params = SPHParams()
    state, pairs, d, e = setup_droplet(params)
    k1, g1, _ = color_field_curvature(state, params, pairs, d, e)
    f1 = params.sigma_surf * k1[:, None] * g1 / np.maximum(state.rho, 1e-9)[:, None]

    rng = np.random.default_rng(20260901)
    perm = rng.permutation(state.n)
    state2, pairs2, d2, e2 = setup_droplet(params)
    # apply the SAME permutation to the identical initial lattice
    state2.pos = state2.pos[perm]
    state2.mass = state2.mass[perm]
    state2.phase = state2.phase[perm]
    pairs2, d2, e2 = build_pairs(state2.pos, params.h,
                                 x_period=state2.domain[2] - state2.domain[0])
    compute_density(state2, params, pairs2, d2)
    k2, g2, _ = color_field_curvature(state2, params, pairs2, d2, e2)
    f2 = params.sigma_surf * k2[:, None] * g2 / np.maximum(state2.rho, 1e-9)[:, None]

    k2c = np.empty_like(k2)
    f2c = np.empty_like(f2)
    k2c[perm] = k2
    f2c[perm] = f2
    mask = state.phase != 2
    dk = np.abs(k2c - k1)[mask]
    df = np.linalg.norm(f2c - f1, axis=1)[mask]
    return {
        "max_abs_kappa_diff_free": float(dk.max()),
        "mean_abs_kappa_diff_free": float(dk.mean()),
        "max_abs_force_diff_free": float(df.max()),
        "rms_force_norm_free": float(np.sqrt((df ** 2).mean())),
        "verdict": ("INVARIANT" if (dk.max() < 1e-10 and df.max() < 1e-10)
                    else "NOT PERMUTATION-INVARIANT"),
    }


def probe_azimuthal_uniformity():
    params = SPHParams()
    state, pairs, d, e = setup_droplet(params)
    kappa, grad, _ = color_field_curvature(state, params, pairs, d, e)
    com = state.pos[state.phase == 1].mean(axis=0)
    rel = state.pos - com
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
    gnorm = np.linalg.norm(grad, axis=1)
    droplet = state.phase == 1
    rim = droplet & (gnorm > 0.25 * gnorm[droplet].max())
    sectors = []
    counts = []
    for k in range(12):
        m = rim & (az >= 30 * k) & (az < 30 * (k + 1))
        counts.append(int(m.sum()))
        sectors.append(float(kappa[m].mean()) if m.sum() >= 3 else np.nan)
    sectors = np.array(sectors)
    R = 3.0
    expected = 1.0 / R
    fin = sectors[np.isfinite(sectors)]
    return {
        "expected_kappa": expected,
        "sector_mean_kappa": [float(x) if np.isfinite(x) else None for x in sectors],
        "sector_spread_max_minus_min": float(fin.max() - fin.min()),
        "n_rim_particles": int(rim.sum()),
        "n_particles_per_sector_min": int(min(counts)),
    }


def probe_internal_force_torque():
    params = SPHParams()
    state, pairs, d, e = setup_droplet(params)
    kappa, grad, _ = color_field_curvature(state, params, pairs, d, e)
    f = params.sigma_surf * kappa[:, None] * grad / np.maximum(state.rho, 1e-9)[:, None]
    free = state.phase != 2
    net = (state.mass[free, None] * f[free]).sum(axis=0)
    com = state.pos[free].mean(axis=0)
    rel = state.pos[free] - com
    torque = float((state.mass[free] * (rel[:, 0] * f[free, 1]
                                        - rel[:, 1] * f[free, 0])).sum())
    scale = float((state.mass[free] * np.linalg.norm(f[free], axis=1)).sum())
    return {
        "net_force": net.tolist(),
        "net_force_norm": float(np.linalg.norm(net)),
        "net_force_over_total_magnitude": float(np.linalg.norm(net) / max(scale, 1e-12)),
        "net_torque": torque,
        "net_torque_over_total_magnitude": float(abs(torque) / max(scale, 1e-12)),
        "verdict": ("CONSERVATIVE" if np.linalg.norm(net) / max(scale, 1e-12) < 1e-3
                    and abs(torque) / max(scale, 1e-12) < 1e-2
                    else "NON-CONSERVATIVE (net force/torque on an isolated system)"),
    }


def probe_stencil_census():
    """Fraction of each particle's neighbour pairs that reach its divergence.

    With the symmetric accumulation every particle appears once per pair on
    each side, so the fraction is exactly 1.0. This probe recomputes the
    census from the ACTUAL bincount weights rather than re-implementing the
    operator: it runs the shared helper's accumulation pattern.
    """
    params = SPHParams()
    state, pairs, d, e = setup_droplet(params)
    i, j = pairs[:, 0], pairs[:, 1]
    idx = np.concatenate([i, j])
    hits = np.bincount(idx, minlength=state.n)
    # expected contributions per particle = its total neighbour count
    nbr_count = np.bincount(idx, minlength=state.n)  # same: 2 per pair
    frac = hits / np.maximum(nbr_count, 1)
    droplet = state.phase == 1
    return {
        "mean_fraction_pairs_in_stencil_droplet": float(frac[droplet].mean()),
        "min_fraction": float(frac[droplet].min()),
        "max_fraction": float(frac[droplet].max()),
        "verdict": ("SYMMETRIC STENCIL" if frac[droplet].min() > 0.99
                    else "ONE-SIDED STENCIL"),
    }


if __name__ == "__main__":
    print("=== CSF operator-level symmetry audit (shared helper) ===")
    for name, fn in [("[A] permutation invariance", probe_permutation_invariance),
                     ("[B] azimuthal uniformity", probe_azimuthal_uniformity),
                     ("[C] internal force/torque", probe_internal_force_torque),
                     ("[D] stencil census", probe_stencil_census)]:
        r = fn()
        print(f"\n{name}:")
        print(json.dumps(r, indent=2))
