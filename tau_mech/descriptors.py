"""Per-conformation descriptors assembled from the lower-level modules.

One call to :func:`compute_model_descriptors` produces every descriptor for a
single conformation; the pipeline loops over conformers and collects them.

Descriptors (all in Angstrom / Angstrom^2 unless noted):
  * radius of gyration (mass-weighted and equal-weight, heavy atoms)
  * end-to-end distance (CA_1 -> CA_n)
  * per-residue SASA and relative SASA (rASA)
  * mean rASA over each aggregation-prone region (VQIINK, VQIVYK)
  * heavy-atom contact map (residue level) and per-residue neighbor counts
  * graph summary: number of edges, mean degree, graph density
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .constants import BACKBONE_ATOMS
from .features import GraphConfig, build_residue_graph
from .geometry import (
    end_to_end_distance,
    heavy_atom_contact_map,
    masses_from_elements,
    per_residue_ca_coords,
    per_residue_neighbor_counts,
    radius_of_gyration,
    side_chain_centroids,
)
from .sasa import compute_sasa, relative_sasa, residue_sasa, residue_rsa_stats, vdw_radii_for_elements
from .numbering import residue_index_from_atoms


def compute_model_descriptors(
    model: Dict,
    one_letter: np.ndarray,
    n_res: int,
    tau_resseq: np.ndarray,
    motif_spans: Dict[str, tuple],
    graph_cfg: Optional[GraphConfig] = None,
    probe: float = 1.4,
    n_probe_points: int = 480,
    contact_cutoff: float = 5.0,
    neighbor_cutoff: float = 8.0,
) -> Dict:
    """Compute all descriptors for a single conformation (dict of arrays)."""
    coords = model["coords"].astype(np.float64)
    elements = [str(e) for e in model["element"]]
    atom_res_idx = residue_index_from_atoms(model["resseq"], model["chain"])
    masses = masses_from_elements(elements)

    # --- geometry --------------------------------------------------------
    rg_mass = radius_of_gyration(coords, masses, mass_weighted=True)
    rg_equal = radius_of_gyration(coords, mass_weighted=False)
    ca = per_residue_ca_coords(coords, atom_res_idx, model["name"], n_res)
    e2e = end_to_end_distance(ca)

    # --- SASA --------------------------------------------------------------
    radii = vdw_radii_for_elements(elements)
    atom_sasa = compute_sasa(coords, radii, probe=probe, n_points=n_probe_points)
    res_sasa = residue_sasa(atom_sasa, atom_res_idx, n_res)
    res_rsa = relative_sasa(res_sasa, one_letter)
    apr = residue_rsa_stats(res_rsa, motif_spans, tau_resseq)

    # --- contacts / burial --------------------------------------------------
    contact_map = heavy_atom_contact_map(coords, atom_res_idx, n_res, cutoff=contact_cutoff)
    n_contacts = int(contact_map.sum() // 2)
    neighbor_counts = per_residue_neighbor_counts(coords, atom_res_idx, n_res, cutoff=neighbor_cutoff)

    # --- graph --------------------------------------------------------------
    sc = side_chain_centroids(coords, atom_res_idx, model["name"], n_res, BACKBONE_ATOMS)
    graph = build_residue_graph(
        coords, atom_res_idx, n_res, one_letter, ca, sc, model["name"], graph_cfg,
    )
    n_edges = graph["edge_index"].shape[1]
    mean_degree = (2.0 * n_edges / n_res) if n_res else 0.0
    density = (2.0 * n_edges / (n_res * (n_res - 1))) if n_res > 1 else 0.0

    return {
        "rg_mass_weighted": rg_mass,
        "rg_equal_weight": rg_equal,
        "end_to_end": e2e,
        "res_sasa": res_sasa.astype(np.float32),
        "res_rsa": res_rsa.astype(np.float32),
        "apr_mean_rsa": np.asarray(
            [apr.get("VQIINK", np.nan), apr.get("VQIVYK", np.nan)], dtype=np.float32
        ),
        "contact_map": contact_map,
        "n_contacts": n_contacts,
        "neighbor_counts": neighbor_counts.astype(np.int16),
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "graph_density": density,
        "graph": graph,
    }
