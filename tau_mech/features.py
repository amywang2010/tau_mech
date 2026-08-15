"""Residue-level geometric graph construction for the Tau ensembles.

Graph schema (residue-level):
  * one node per residue of the protein chain
  * node features: 21-dim one-hot amino acid code, Kyte-Doolittle hydropathy,
    normalized sequence position (concatenated -> 23-dim)
  * node spatial positions: C-alpha coordinate and side-chain centroid
  * edges: (a) spatial edges -- any pair of residues with a heavy-atom pair
    closer than the cutoff (min heavy-atom distance criterion), and
    (b) optional sequential edges connecting residues i..i+k (chain
    connectivity)
  * edge attributes: [min heavy-atom distance (A), sequence separation |i-j|]

The schema is stored as plain numpy arrays (per-model .npz) so it can be
consumed by any downstream framework (PyTorch Geometric, networkx, ...)
without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .constants import (
    AA_TO_INDEX,
    BACKBONE_ATOMS,
    DEFAULT_EDGE_CUTOFF,
    HYDROPATHY,
    SEQ_ADJACENCY,
    UNKNOWN_INDEX,
)


@dataclass
class GraphConfig:
    """Graph construction parameters (serialized into the output config)."""
    edge_cutoff: float = DEFAULT_EDGE_CUTOFF     # heavy-atom distance cutoff (A)
    add_sequential: bool = True                  # connect residues i..i+SEQ_ADJACENCY
    seq_adjacency: int = SEQ_ADJACENCY           # max sequence separation for seq edges


def build_residue_graph(
    coords: np.ndarray,
    atom_res_idx: np.ndarray,
    n_res: int,
    res_names_1: Sequence[str],
    ca_coords: np.ndarray,
    sc_centroids: np.ndarray,
    atom_names: Sequence[str],
    cfg: Optional[GraphConfig] = None,
) -> Dict[str, np.ndarray]:
    """Build a residue-level graph for one conformation.

    Parameters
    ----------
    coords : (A, 3) heavy-atom coordinates
    atom_res_idx : (A,) residue index per atom
    n_res : number of residues
    res_names_1 : one-letter codes per residue
    ca_coords : (n_res, 3) C-alpha coordinates
    sc_centroids : (n_res, 3) side-chain centroids
    atom_names : atom names per atom (for backbone/side-chain split)
    cfg : GraphConfig

    Returns a dict with keys:
        num_nodes, node_features (n_res, 23), node_pos_ca (n_res, 3),
        node_pos_sc (n_res, 3), edge_index (2, E), edge_attr (E, 2)
    """
    cfg = cfg or GraphConfig()
    coords = np.asarray(coords, dtype=np.float64)

    # --- node features ---------------------------------------------------
    n = n_res
    feats = np.zeros((n, 23), dtype=np.float32)
    for i, aa in enumerate(res_names_1):
        aa = str(aa).upper()
        feats[i, AA_TO_INDEX.get(aa, UNKNOWN_INDEX)] = 1.0
        feats[i, 21] = HYDROPATHY.get(aa, 0.0)
        feats[i, 22] = i / max(n - 1, 1)
    node_pos_ca = np.asarray(ca_coords, dtype=np.float32)
    node_pos_sc = np.asarray(sc_centroids, dtype=np.float32)

    # --- spatial edges (min heavy-atom distance < cutoff) ----------------
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(cfg.edge_cutoff), output_type="ndarray")
    edge_i = []
    edge_j = []
    edge_d = []
    if len(pairs):
        ri = atom_res_idx[pairs[:, 0]]
        rj = atom_res_idx[pairs[:, 1]]
        keep = ri != rj
        pairs = pairs[keep]
        ri = ri[keep]
        rj = rj[keep]
        d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        # keep the minimum distance per unordered residue pair
        key = np.sort(np.column_stack([ri, rj]), axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        edge_i = ri[first].astype(np.int64)
        edge_j = rj[first].astype(np.int64)
        edge_d = d[first]

    if cfg.add_sequential:
        seq_i, seq_j, seq_d = [], [], []
        for k in range(1, cfg.seq_adjacency + 1):
            a = np.arange(0, n - k)
            seq_i.append(a)
            seq_j.append(a + k)
            seq_d.append(np.full(len(a), float("nan")))
        edge_i = np.concatenate([edge_i] + seq_i).astype(np.int64)
        edge_j = np.concatenate([edge_j] + seq_j).astype(np.int64)
        edge_d = np.concatenate([edge_d] + seq_d).astype(np.float64)

    # dedupe (a spatial edge may duplicate a sequential edge)
    key = np.sort(np.column_stack([edge_i, edge_j]), axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    edge_i = edge_i[first]
    edge_j = edge_j[first]
    edge_d = edge_d[first]

    # --- edge attributes -------------------------------------------------
    seq_sep = np.abs(edge_i - edge_j).astype(np.float32)
    edge_attr = np.column_stack([edge_d, seq_sep]).astype(np.float32)
    edge_index = np.stack([edge_i, edge_j]).astype(np.int32)

    return {
        "num_nodes": n,
        "node_features": feats,
        "node_pos_ca": node_pos_ca,
        "node_pos_sc": node_pos_sc,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
