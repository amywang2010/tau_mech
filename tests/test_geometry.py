"""Tests for geometric descriptors (tau_mech.geometry)."""

import numpy as np
import pytest

from tau_mech.geometry import (
    end_to_end_distance,
    heavy_atom_contact_map,
    inter_residue_min_distance,
    masses_from_elements,
    per_residue_ca_coords,
    per_residue_neighbor_counts,
    radius_of_gyration,
    side_chain_centroids,
)


def test_radius_of_gyration_two_points():
    coords = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    assert radius_of_gyration(coords, mass_weighted=False) == pytest.approx(1.0)
    # mass-weighted with equal masses is the same
    m = np.array([1.0, 1.0])
    assert radius_of_gyration(coords, m, mass_weighted=True) == pytest.approx(1.0)


def test_radius_of_gyration_mass_weighted():
    coords = np.array([[2.0, 0, 0], [0.0, 0, 0]])
    masses = np.array([3.0, 1.0])
    # center of mass at x=1.5; distances: 0.5 and 1.5
    # Rg^2 = (3*0.25 + 1*2.25)/4 = (0.75+2.25)/4 = 0.75
    assert radius_of_gyration(coords, masses, mass_weighted=True) == pytest.approx(
        np.sqrt(0.75)
    )


def test_masses_from_elements():
    m = masses_from_elements(["C", "N", "O", "S"])
    assert m[0] == pytest.approx(12.011)
    assert m[1] == pytest.approx(14.007)
    assert m[2] == pytest.approx(15.999)
    assert m[3] == pytest.approx(32.06)


def test_end_to_end_distance():
    ca = np.array([[0.0, 0, 0], [0, 0, 0], [3.0, 4.0, 0]])
    assert end_to_end_distance(ca) == pytest.approx(5.0)


def _three_residue_system():
    """3 residues; res1 at (0,0,0), res2 at (2.5,0,0), res3 at (10,0,0)."""
    coords = np.array([
        [0.0, 0.0, 0.0],    # r0 atom0
        [0.0, 1.0, 0.0],    # r0 atom1
        [2.5, 0.0, 0.0],    # r1 atom0
        [2.6, 0.0, 0.0],    # r1 atom1
        [10.0, 0.0, 0.0],   # r2 atom0
        [10.0, 1.0, 0.0],   # r2 atom1
    ])
    res_idx = np.array([0, 0, 1, 1, 2, 2])
    names = np.array(["N", "CA", "N", "CA", "N", "CA"])
    return coords, res_idx, names


def test_contact_map():
    coords, res_idx, _ = _three_residue_system()
    cm = heavy_atom_contact_map(coords, res_idx, n_res=3, cutoff=5.0)
    assert cm[0, 1] == 1 and cm[1, 0] == 1   # close pair
    assert cm[0, 2] == 0 and cm[2, 0] == 0   # far pair
    assert np.all(np.diag(cm) == 0)


def test_neighbor_counts():
    coords, res_idx, _ = _three_residue_system()
    # r0<->r1 (0->2.6 A) and r1<->r2 (2.6->10 A = 7.4 A) are within 8 A;
    # r0<->r2 (10 A) is not.
    counts = per_residue_neighbor_counts(coords, res_idx, n_res=3, cutoff=8.0)
    assert counts.tolist() == [1, 2, 1]


def test_min_distance_sparse():
    coords, res_idx, _ = _three_residue_system()
    md = inter_residue_min_distance(coords, res_idx, n_res=3, cutoff=5.0)
    assert md.shape[0] == 1  # only the close pair
    assert sorted(md[0, :2].astype(int).tolist()) == [0, 1]
    assert md[0, 2] == pytest.approx(2.5)


def test_ca_and_side_chain_centroids():
    coords, res_idx, names = _three_residue_system()
    ca = per_residue_ca_coords(coords, res_idx, names, n_res=3)
    assert ca[0, 0] == pytest.approx(0.0)
    assert ca[1, 0] == pytest.approx(2.6)
    sc = side_chain_centroids(coords, res_idx, names, n_res=3,
                              backbone_atoms={"N", "CA", "C", "O"})
    # residue 0: only N and CA -> no side chain -> falls back to CA
    np.testing.assert_allclose(sc[0], ca[0])
