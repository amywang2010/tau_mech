"""Regression tests for the CSF interface operators (2026-09-02 fix).

The original divergence accumulation was one-sided (only the first pair
endpoint received the term), producing a label-dependent, non-conservative
curvature field: NOT permutation-invariant (kappa moved by up to 1.5 under a
pure relabeling), azimuthally biased on a circle, and with a net internal
surface force of 10.8% of its own magnitude. These tests lock in the
symmetric form.
"""
import numpy as np
import pytest

from tau_mech.sph import (
    SPHParams,
    build_pairs,
    color_field_curvature,
    compute_density,
    make_couette_droplet_state,
    run,
)


def _setup(params, domain=(0.0, 0.0, 24.0, 16.0), radius=3.0):
    state = make_couette_droplet_state(params, domain=domain, spacing=0.5,
                                       droplet_radius=radius, n_wall_layers=4)
    run(state, params, 0, 0.008)
    pairs, d, e = build_pairs(state.pos, params.h,
                              x_period=state.domain[2] - state.domain[0])
    compute_density(state, params, pairs, d)
    return state, pairs, d, e


def _surface_accel(params, state, pairs, d, e):
    kappa, grad, _ = color_field_curvature(state, params, pairs, d, e)
    return (params.sigma_surf * kappa[:, None] * grad
            / np.maximum(state.rho, 1e-9)[:, None])


def test_csf_permutation_invariance():
    """kappa and the surface acceleration cannot depend on particle labels."""
    p = SPHParams()
    for seed in (7, 20260901):
        state, pairs, d, e = _setup(p)
        k1 = color_field_curvature(state, p, pairs, d, e)[0]
        f1 = _surface_accel(p, state, pairs, d, e)

        perm = np.random.default_rng(seed).permutation(state.n)
        state2, pairs2, d2, e2 = _setup(p)
        state2.pos = state2.pos[perm]
        state2.mass = state2.mass[perm]
        state2.phase = state2.phase[perm]
        pairs2, d2, e2 = build_pairs(state2.pos, p.h,
                                     x_period=state2.domain[2] - state2.domain[0])
        compute_density(state2, p, pairs2, d2)
        k2 = color_field_curvature(state2, p, pairs2, d2, e2)[0]
        f2 = _surface_accel(p, state2, pairs2, d2, e2)

        # un-permute: canonical_results[i] = permuted_results[perm[i]]
        k2c = np.empty_like(k2)
        f2c = np.empty_like(f2)
        k2c[perm] = k2
        f2c[perm] = f2
        mask = state.phase != 2
        assert np.abs(k2c - k1)[mask].max() == pytest.approx(0.0, abs=1e-10)
        df = np.linalg.norm(f2c - f1, axis=1)[mask]
        assert df.max() == pytest.approx(0.0, abs=1e-10)


def test_csf_internal_force_conservative():
    """The internal surface force must have (near-)zero net force and torque."""
    p = SPHParams()
    state, pairs, d, e = _setup(p)
    f = _surface_accel(p, state, pairs, d, e)
    free = state.phase != 2
    net = (state.mass[free, None] * f[free]).sum(axis=0)
    scale = float((state.mass[free]
                   * np.linalg.norm(f[free], axis=1)).sum())
    assert np.linalg.norm(net) / scale < 1e-3
    com = state.pos[free].mean(axis=0)
    rel = state.pos[free] - com
    torque = (state.mass[free] * (rel[:, 0] * f[free, 1]
                                  - rel[:, 1] * f[free, 0])).sum()
    assert abs(torque) / scale < 1e-2


def test_csf_azimuthal_spread_bounded():
    """On a circle the rim curvature must be near-uniform in azimuth.

    The R=3 droplet is coarse (h/R = 1/3, documented band ~3h); the
    symmetric stencil reduced the sector spread by 3x vs the one-sided
    stencil. The bound here is the POST-FIX measured value + margin; a
    regression of the stencil symmetry reproduces the old 0.33 spread.
    """
    p = SPHParams()
    state, pairs, d, e = _setup(p)
    kappa, grad, _ = color_field_curvature(state, p, pairs, d, e)
    com = state.pos[state.phase == 1].mean(axis=0)
    rel = state.pos - com
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
    gnorm = np.linalg.norm(grad, axis=1)
    droplet = state.phase == 1
    rim = droplet & (gnorm > 0.25 * gnorm[droplet].max())
    sectors = []
    for k in range(12):
        m = rim & (az >= 30 * k) & (az < 30 * (k + 1))
        if m.sum() >= 3:
            sectors.append(float(kappa[m].mean()))
    spread = max(sectors) - min(sectors)
    assert spread < 0.15  # post-fix measured 0.106; one-sided was 0.331
