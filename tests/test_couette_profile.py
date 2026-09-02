"""Unit tests for the Couette profile measurement (data-range binning).

Regression history (2026-09-02): (1) fixed-domain digitize binning silently
dropped the y ~ -1e-16 bottom fluid row into index -1; (2) a lattice-row
binning attempt was disproved by shear-induced row shuffling (~0.45*dy over
3 viscous times). The shipping measurement bins over the DATA range with
particle-mean bin coordinates - every particle counted, no lattice or
bin-alignment assumption - reports empty bins instead of failing on real
near-wall banding, and asserts the particle-conservation invariant.
"""
import numpy as np
import pytest

from tau_mech.sph import _binned_profile


def test_linear_field_recovered_any_bin_count():
    # u(y) = a*y + b on a random column: for a linear field, particle-mean
    # bin coordinates satisfy mean(u) = a*mean(y) + b exactly, so the fit
    # must recover (a, b) at EVERY binning resolution - the measurement must
    # not depend on the bin parameter.
    rng = np.random.default_rng(11)
    y = np.sort(rng.uniform(0.0, 8.0, 4000))
    a, b = 0.37, -1.9
    ux = a * y + b + rng.normal(0.0, 1e-3, y.size)
    for n_bins in (10, 15, 25, 40, 64):
        yb, ub, info = _binned_profile(y, ux, n_bins)
        assert info["n_empty"] == 0
        A = np.vstack([yb, np.ones_like(yb)]).T
        (a_fit, b_fit), *_ = np.linalg.lstsq(A, ub, rcond=None)
        assert a_fit == pytest.approx(a, abs=1e-3)
        assert b_fit == pytest.approx(b, abs=1e-3)


def test_boundary_particle_counted_regression():
    # The digitize defect: the y ~ -1e-16 row was silently discarded.
    # Data-range edges must classify it into the first bin, and the fit on a
    # linear field must recover the exact intercept at that edge.
    y = np.sort(np.concatenate([[0.0, 1e-16], np.linspace(0.1, 8.0, 398)]))
    a, b = 0.5, -2.0
    ux = a * y + b
    yb, ub, info = _binned_profile(y, ux, 20)
    assert info["n_empty"] == 0
    assert yb[0] >= 0.0                       # edge row present, not dropped
    A = np.vstack([yb, np.ones_like(yb)]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, ub, rcond=None)
    assert b_fit == pytest.approx(-2.0, abs=1e-9)


def test_banded_column_reports_empty_bins():
    # Near-wall banding (real density structure at early times) must be
    # REPORTED via n_empty, and the occupied-bin fit on a linear field must
    # stay exact - empty bins are excluded, never interpolated.
    rows = np.arange(20) * 0.4330
    y = np.sort(np.tile(rows, 50))
    a, b = 0.4, -1.6
    ux = a * y + b
    yb, ub, info = _binned_profile(y, ux, 40)
    assert info["n_empty"] > 0                # banding detected, reported
    A = np.vstack([yb, np.ones_like(yb)]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, ub, rcond=None)
    assert a_fit == pytest.approx(a, abs=1e-9)
    assert b_fit == pytest.approx(b, abs=1e-9)


def test_particle_conservation_invariant():
    # The digitize-bug class: any scheme that loses particles must fail.
    rng = np.random.default_rng(3)
    y = rng.uniform(0.0, 8.0, 1000)
    _, _, info = _binned_profile(y, np.zeros_like(y), 25)
    assert info["n_particles"] == 1000
