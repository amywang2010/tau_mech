"""Tests for residue-level graph construction (tau_mech.features)."""

import numpy as np
import pytest

from tau_mech.features import GraphConfig, build_residue_graph


def _synthetic_system():
    """3 residues in a line; r0-r1 close, r2 far."""
    coords = np.array([
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],    # r0: N, CA
        [2.5, 0.0, 0.0], [2.6, 0.0, 0.0],    # r1: N, CA
        [10.0, 0.0, 0.0], [10.0, 1.0, 0.0],  # r2: N, CA
    ])
    res_idx = np.array([0, 0, 1, 1, 2, 2])
    names = np.array(["N", "CA", "N", "CA", "N", "CA"])
    ca = np.array([[0.0, 1.0, 0.0], [2.6, 0.0, 0.0], [10.0, 1.0, 0.0]])
    sc = ca.copy()
    return coords, res_idx, names, ca, sc


def test_node_features():
    coords, res_idx, names, ca, sc = _synthetic_system()
    g = build_residue_graph(coords, res_idx, 3, np.asarray(["A", "G", "V"]),
                            ca, sc, names, GraphConfig())
    assert g["node_features"].shape == (3, 23)
    # one-hot sum to 1 per node
    np.testing.assert_allclose(g["node_features"][:, :21].sum(axis=1), 1.0)


def test_spatial_and_sequential_edges():
    coords, res_idx, names, ca, sc = _synthetic_system()
    g = build_residue_graph(coords, res_idx, 3, np.asarray(["A", "G", "V"]),
                            ca, sc, names, GraphConfig())
    ei = g["edge_index"]
    pairs = set(map(tuple, ei.T.tolist()))
    # sequential edges: (0,1), (1,2) [adjacency 2 also gives (0,2)]
    assert (0, 1) in pairs and (1, 2) in pairs
    # no self loops
    assert all(i != j for i, j in pairs)
    # spatial edge (0,1) present
    assert (0, 1) in pairs
    # edge attributes: [min distance, seq sep]
    ea = g["edge_attr"]
    assert ea.shape[0] == ei.shape[1]
    assert ea.shape[1] == 2


def test_no_sequential_edges():
    coords, res_idx, names, ca, sc = _synthetic_system()
    g = build_residue_graph(coords, res_idx, 3, np.asarray(["A", "G", "V"]),
                            ca, sc, names,
                            GraphConfig(add_sequential=False))
    ei = g["edge_index"]
    pairs = set(map(tuple, ei.T.tolist()))
    assert (0, 1) in pairs          # spatial
    assert (1, 2) not in pairs      # no sequential edge for far pair


def test_edges_deduped():
    """Sequential + spatial duplicate edges must appear once."""
    coords, res_idx, names, ca, sc = _synthetic_system()
    g = build_residue_graph(coords, res_idx, 3, np.asarray(["A", "G", "V"]),
                            ca, sc, names, GraphConfig())
    ei = g["edge_index"]
    key = np.sort(ei, axis=0).T
    assert len(np.unique(key, axis=0)) == len(key)
