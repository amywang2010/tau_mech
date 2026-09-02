"""Unit tests for the lattice-row Couette profile measurement.

Regression coverage for the fixed-width digitize binning defect (2026-09-02):
digitize silently dropped the y ~ -1e-16 bottom fluid row into index -1 and
made the wall-slip reading dependent on bin-width/lattice alignment. The
lattice-row helper must (a) keep every fluid row, (b) recover exact row means
for a linear field, (c) report empty rows instead of failing silently.
"""
import numpy as np
import pytest

from tau_mech.sph import _wall_lattice, _lattice_row_profile


def _lattice(H):
    pos, fluid, inner_bottom, inner_top = _wall_lattice(0, 0, 24, H, 0.5, 4)
    return pos[fluid, 1].copy(), inner_bottom, inner_top


def test_linear_profile_exact_row_means():
    # u(y) = a*y + b evaluated on the real lattice, plus sub-row jitter:
    # row means must reproduce (a, b) to jitter scale, not bin-alignment scale.
    y, ib, it = _lattice(8.0)
    dy = (it - ib) / (round((it - ib) / (0.5 * np.sqrt(3) / 2)))  # row spacing
    a, b = 0.37, -1.9
    rng = np.random.default_rng(7)
    jitter = rng.normal(0.0, 1e-3, len(y)) * dy
    ux = a * (y + jitter) + b
    row_y, u_row, n_empty, max_dev = _lattice_row_profile(
        y + jitter, ux, ib, 0.5 * np.sqrt(3) / 2)
    assert n_empty == 0
    # rows must span the full fluid column: bottom row at ~0 (NOT dropped)
    assert row_y.min() == pytest.approx(0.0, abs=1e-9)
    assert row_y.max() == pytest.approx(it - dy, abs=1e-9)
    # linear fit recovers the field exactly up to jitter averaging
    A = np.vstack([row_y, np.ones_like(row_y)]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, u_row, rcond=None)
    assert a_fit == pytest.approx(a, abs=1e-3)
    assert b_fit == pytest.approx(b, abs=1e-3)
    assert max_dev < 1e-2 * dy


def test_no_boundary_row_drop_regression():
    # The defect: digitize placed the y ~ -1e-16 row outside the bins.
    # The helper must assign it to the first lattice row.
    y, ib, _ = _lattice(8.0)
    y = y.copy()
    y[0] = -1.1102230246251565e-16  # the documented accumulator artifact
    ux = -2.0 * np.ones(len(y))     # constant field: row mean = -2
    row_y, u_row, n_empty, _ = _lattice_row_profile(
        y, ux, ib, 0.5 * np.sqrt(3) / 2)
    assert n_empty == 0
    assert row_y[0] == pytest.approx(0.0, abs=1e-9)
    assert u_row[0] == pytest.approx(-2.0)


def test_empty_row_is_reported_not_silent():
    # Remove every particle of one interior row: the helper must report it
    # (validate_couette raises on this) rather than interpolate silently.
    y, ib, it = _lattice(8.0)
    dy = 0.5 * np.sqrt(3) / 2
    k = np.round((y - ib) / dy).astype(int)
    kill = k == (k.min() + 5)
    keep = ~kill
    row_y, u_row, n_empty, _ = _lattice_row_profile(
        y[keep], np.zeros(keep.sum()), ib, dy)
    assert n_empty >= 1
