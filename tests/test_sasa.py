"""Tests for the Shrake-Rupley SASA implementation (tau_mech.sasa)."""

import numpy as np
import pytest

from tau_mech.constants import VDW_RADII
from tau_mech.sasa import (
    compute_sasa,
    fibonacci_sphere,
    relative_sasa,
    residue_sasa,
    residue_rsa_stats,
    vdw_radii_for_elements,
)


def test_fibonacci_sphere_unit():
    pts = fibonacci_sphere(480)
    assert pts.shape == (480, 3)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_single_atom_sphere_sasa():
    """One isolated atom: SASA = 4*pi*(r+probe)^2."""
    coords = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([VDW_RADII["C"]])
    sasa = compute_sasa(coords, radii, probe=1.4, n_points=2000)
    expected = 4 * np.pi * (1.8 + 1.4) ** 2
    assert sasa[0] == pytest.approx(expected, rel=0.02)


def test_self_not_occluding_own_probe_points():
    """Regression test: an atom must not occlude its own probe points.

    A probe point sits exactly ON atom i's occlusion sphere (radius
    radii[i] + probe), so under the strict `<` boundary convention it must NOT
    count as occluded by i itself. Floating-point rounding of the Fibonacci
    unit vectors (|pts| slightly < 1 for ~22% of the points) used to flip
    those boundary points into false self-occlusion. For a FULLY exposed atom
    this under-estimated SASA by ~10% (89.4% -> 99.4%); for partially buried
    atoms the relative error is much larger (a ~30%-exposed residue is roughly
    halved), which is why the APR rASA doubled after the fix (audit
    2026-08-15). Here atom 0 has a neighbor at 6.3 A — inside the 6.4 A query
    radius so the neighbor-occlusion path runs, but with only ~0.6% genuine
    occlusion — so its SASA must stay above 95% of the full sphere area (the
    buggy code gave 89.4%).
    """
    r = VDW_RADII["C"]
    probe = 1.4
    full = 4 * np.pi * (r + probe) ** 2
    coords = np.array([[0.0, 0.0, 0.0], [6.3, 0.0, 0.0]])
    radii = np.array([r, r])
    sasa = compute_sasa(coords, radii, probe=probe, n_points=480)
    assert sasa[0] > 0.95 * full


def test_two_atoms_occlusion():
    """Two overlapping atoms have combined SASA < sum of individual SASAs."""
    r = VDW_RADII["C"]
    single = compute_sasa(np.zeros((1, 3)), np.array([r]), probe=1.4, n_points=1000)[0]
    d = 1.5  # well inside 2*(r+1.4) = 6.4 A -> substantial overlap
    coords = np.array([[0, 0, 0], [d, 0, 0]])
    radii = np.array([r, r])
    sasa = compute_sasa(coords, radii, probe=1.4, n_points=1000)
    assert sasa.sum() < 2 * single
    assert sasa.sum() > single  # still more than one atom's surface


def test_sasa_order_invariance():
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(20, 3))
    radii = rng.uniform(1.4, 1.9, size=20)
    s1 = compute_sasa(coords, radii, probe=1.4, n_points=480)
    perm = rng.permutation(20)
    s2 = compute_sasa(coords[perm], radii[perm], probe=1.4, n_points=480)
    np.testing.assert_allclose(s1[perm], s2, rtol=1e-10)


def test_vdw_radii_mapping():
    radii = vdw_radii_for_elements(["C", "N", "O", "S", "?"])
    np.testing.assert_allclose(
        radii, [1.80, 1.65, 1.40, 1.85, 1.80], atol=1e-6
    )


def test_residue_sasa_sum():
    atom_sasa = np.array([1.0, 2.0, 3.0, 4.0])
    atom_res = np.array([0, 0, 1, 1])
    rs = residue_sasa(atom_sasa, atom_res, n_res=2)
    np.testing.assert_allclose(rs, [3.0, 7.0])


def test_relative_sasa_bounds():
    """rASA = SASA / reference; the reference is the Tien et al. 2013 max-ASA
    table from constants. A residue in a less-crowded conformation than the
    Gly-X-Gly reference tripeptide can modestly exceed 1.0 (the tripeptide
    context itself occludes part of the side chain), so the exact ratio
    against the recorded reference is asserted rather than a hard 1.0 cap.
    """
    from tau_mech.constants import REFERENCE_SASA
    rs = relative_sasa(np.array([200.0, 5.0]), np.asarray(["A", "G"]))
    assert rs[0] == pytest.approx(200.0 / REFERENCE_SASA["A"])
    assert rs[1] == pytest.approx(5.0 / REFERENCE_SASA["G"])
    assert rs[0] < 2.0  # sane magnitude; exact ratio recorded above


def test_residue_rsa_stats():
    rsa = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.5])
    tau = np.array([274, 275, 276, 277, 278, 279, 280])
    out = residue_rsa_stats(rsa, {"VQIINK": (275, 280)}, tau)
    assert out["VQIINK"] == pytest.approx(np.mean([0.1, 0.8, 0.2, 0.7, 0.3, 0.5]))
