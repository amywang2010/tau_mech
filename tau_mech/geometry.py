"""Geometric descriptors of protein conformations.

Implements ensemble-level and per-model geometry descriptors used for EDA and
as graph/node features:

  * mass-weighted and equal-weight radius of gyration (Rg)
  * end-to-end distance (CA of first vs last residue)
  * heavy-atom contact map between residues at a given cutoff
  * per-residue heavy-atom neighbor counts (flexibility / burial proxy)

Definitions follow standard polymer physics (Rg is the root-mean-square
distance of atoms from the center of mass). Mass weighting uses the heavy
element masses from :mod:`tau_mech.constants`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .constants import ATOMIC_MASS


def radius_of_gyration(coords: np.ndarray, masses: Optional[np.ndarray] = None,
                       mass_weighted: bool = True) -> float:
    """Radius of gyration (Angstrom) of a coordinate set.

    Parameters
    ----------
    coords : (N, 3) array
    masses : (N,) array, optional
        Per-atom masses. If None and ``mass_weighted`` is True, masses are
        inferred from the coordinate array alone (equal weights); callers that
        have element information should pass explicit masses.
    mass_weighted : bool
        If True use mass-weighted Rg, else equal-weight Rg.

    Notes
    -----
    The equal-weight Rg (mean over atoms of squared distance to the geometric
    center) is the form commonly compared with SAXS-derived values for IDPs;
    the mass-weighted form is physically more exact. Both are reported.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if masses is None or not mass_weighted:
        center = coords.mean(axis=0)
        diff = coords - center
        return float(np.sqrt((diff ** 2).sum(axis=1).mean()))
    masses = np.asarray(masses, dtype=np.float64)
    center = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    diff = coords - center
    return float(np.sqrt(((diff ** 2).sum(axis=1) * masses).sum() / masses.sum()))


def masses_from_elements(elements: Sequence[str]) -> np.ndarray:
    """Map element symbols to atomic masses (unknown -> carbon mass)."""
    return np.asarray([ATOMIC_MASS.get(str(e).upper(), ATOMIC_MASS["C"]) for e in elements],
                      dtype=np.float64)


def end_to_end_distance(ca_coords: np.ndarray) -> float:
    """Distance (Angstrom) between the first and last C-alpha atoms."""
    ca = np.asarray(ca_coords, dtype=np.float64)
    return float(np.linalg.norm(ca[0] - ca[-1]))


def per_residue_ca_coords(coords: np.ndarray, atom_res_idx: np.ndarray,
                          atom_names: Sequence[str], n_res: int) -> np.ndarray:
    """C-alpha coordinates per residue (fall back to the first atom of the
    residue if no CA atom is present, e.g. truncated structures)."""
    ca = np.full((n_res, 3), np.nan, dtype=np.float64)
    names = np.asarray(atom_names)
    for i in range(n_res):
        idx = np.where(atom_res_idx == i)[0]
        if len(idx) == 0:
            continue
        ca_idx = idx[names[idx] == "CA"]
        if len(ca_idx):
            ca[i] = coords[ca_idx[0]]
        else:
            ca[i] = coords[idx[0]]
    return ca


def side_chain_centroids(coords: np.ndarray, atom_res_idx: np.ndarray,
                         atom_names: Sequence[str], n_res: int,
                         backbone_atoms) -> np.ndarray:
    """Geometric centroid of side-chain heavy atoms per residue.

    Residues without side-chain atoms (glycine, or truncated side chains)
    fall back to the C-alpha position. The set of backbone atom names is
    provided by the caller so the definition is explicit.
    """
    names = np.asarray(atom_names)
    centroids = np.full((n_res, 3), np.nan, dtype=np.float64)
    for i in range(n_res):
        idx = np.where(atom_res_idx == i)[0]
        if len(idx) == 0:
            continue
        sc = idx[~np.isin(names[idx], list(backbone_atoms))]
        if len(sc):
            centroids[i] = coords[sc].mean(axis=0)
        else:
            ca_idx = idx[names[idx] == "CA"]
            centroids[i] = coords[ca_idx[0]] if len(ca_idx) else coords[idx[0]]
    return centroids


def heavy_atom_contact_map(coords: np.ndarray, atom_res_idx: np.ndarray,
                           n_res: int, cutoff: float = 5.0) -> np.ndarray:
    """Residue-residue contact map: entry (i, j) = 1 if any heavy atom of
    residue i is within ``cutoff`` Angstrom of any heavy atom of residue j.

    Uses scipy cKDTree.query_pairs, so only the pairs within the cutoff are
    ever formed (efficient for sparse IDP conformers).
    """
    coords = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(cutoff), output_type="ndarray")
    contact = np.zeros((n_res, n_res), dtype=np.uint8)
    if len(pairs):
        ri = atom_res_idx[pairs[:, 0]]
        rj = atom_res_idx[pairs[:, 1]]
        mask = ri != rj
        contact[ri[mask], rj[mask]] = 1
        contact[rj[mask], ri[mask]] = 1
    return contact


def per_residue_neighbor_counts(coords: np.ndarray, atom_res_idx: np.ndarray,
                                n_res: int, cutoff: float = 8.0) -> np.ndarray:
    """Number of DISTINCT other residues with at least one heavy atom within
    ``cutoff`` A, per residue. A cheap, robust local-burial / flexibility
    proxy (counts unique residue neighbors, not atom pairs)."""
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    counts = np.zeros(n_res, dtype=np.int32)
    pairs = tree.query_pairs(r=float(cutoff), output_type="ndarray")
    if len(pairs):
        ri = atom_res_idx[pairs[:, 0]]
        rj = atom_res_idx[pairs[:, 1]]
        mask = ri != rj
        # dedupe to unordered residue pairs, then count each distinct pair once
        key = np.sort(np.column_stack([ri[mask], rj[mask]]), axis=1)
        upairs = np.unique(key, axis=0)
        if len(upairs):
            counts[upairs[:, 0]] += 1
            counts[upairs[:, 1]] += 1
    return counts


def inter_residue_min_distance(coords: np.ndarray, atom_res_idx: np.ndarray,
                               n_res: int, cutoff: float = 8.0
                               ) -> np.ndarray:
    """Minimum heavy-atom distance between each residue pair that comes within
    ``cutoff`` (a sparse matrix as (K, 3): [res_i, res_j, dmin]); pairs farther
    than the cutoff are omitted (the cutoff defines the graph edge criterion,
    so only in-range pairs are needed for graph construction)."""
    coords = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(cutoff), output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    ri = atom_res_idx[pairs[:, 0]]
    rj = atom_res_idx[pairs[:, 1]]
    keep = ri != rj
    out = np.column_stack([ri[keep], rj[keep], d[keep]])
    # dedupe symmetric pairs, keep the min distance per unordered pair
    key = np.sort(out[:, :2], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    return out[first]
