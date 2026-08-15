"""End-to-end preprocessing pipeline.

For each selected ensemble:

  1. stream the PDB directly from the compressed archive (no disk writes)
  2. parse every MODEL into atom arrays (hydrogens dropped by default)
  3. normalize residue names, extract the sequence, resolve Tau numbering,
     and detect the aggregation-prone motifs (VQIINK / VQIVYK)
  4. compute geometry descriptors (Rg, end-to-end), Shrake-Rupley SASA and
     relative SASA, heavy-atom contact map, and the residue-level graph
  5. write per-model .npz graph files, a consolidated .npz of fixed-shape
     arrays, a per-model summary CSV, and a JSON summary

The output directory mirrors the source ensembles so provenance is clear:
    outputs/
      PED00422/models/model_00000.npz   (per-conformer graphs)
      PED00422/ensemble_data.npz        (consolidated fixed-shape arrays)
      PED00422/summary.csv / summary.json
      ...
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Dict, List

import numpy as np

from .config import PipelineConfig
from .constants import AGGREGATION_PRONE_MOTIFS, ENSEMBLES
from .descriptors import compute_model_descriptors
from .features import GraphConfig
from .io import extract_remark_conformer_ids, open_ensemble_pdb, parse_models
from .numbering import normalize_resnames, residue_index_from_atoms, resolve_numbering


def _expected_motifs(ensemble_id: str) -> Dict[str, tuple]:
    """APR positions in Tau-441 numbering expected for this construct, used to
    validate the motifs detected in the file's actual sequence."""
    return dict(AGGREGATION_PRONE_MOTIFS)


def _first_atom_per_residue(model: Dict) -> np.ndarray:
    """Indices of the first atom of each residue (residue-level view of the
    per-atom arrays). Residues are identified by (chain, resseq) pairs."""
    res_idx = residue_index_from_atoms(model["resseq"], model["chain"])
    _, first_atom = np.unique(res_idx, return_index=True)
    return first_atom


def process_ensemble(cfg: PipelineConfig, ensemble_id: str, save_models: bool = True,
                     save_contact_maps: bool = True, progress_every: int = 25) -> dict:
    """Process one ensemble; returns a summary dict."""
    ec = ENSEMBLES[ensemble_id]
    archive = os.path.join(cfg.data_dir, ec["archive"])
    if not os.path.exists(archive):
        raise FileNotFoundError(f"archive not found: {archive}")

    out_dir = os.path.join(cfg.output_dir, ensemble_id)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    t0 = time.time()
    n_cap = cfg.n_models if cfg.n_models else ec["n_expected_models"]
    graph_cfg = cfg.graph if isinstance(cfg.graph, GraphConfig) else GraphConfig(**cfg.graph)        # ---------- conformer-id metadata (REMARK lines, PED00422) ----------
    # Note: this scans the REMARK header only (stops at the first MODEL line),
    # so the second, full decompression below is the only expensive pass.
    conformer_ids: Dict[int, str] = {}
    with open_ensemble_pdb(archive, ec["member"], ec["inner"]) as raw:
        conformer_ids = extract_remark_conformer_ids(io.TextIOWrapper(raw, encoding="ascii"))

    # ---------- main pass --------------------------------------------------
    records: List[dict] = []
    processed = 0

    with open_ensemble_pdb(archive, ec["member"], ec["inner"]) as raw:
        text = io.TextIOWrapper(raw, encoding="ascii")
        gen = parse_models(
            text,
            heavy_only=cfg.heavy_only,
            with_occupancy=cfg.with_occupancy,
            max_models=cfg.n_models,
            start_at=cfg.start_at,
        )
        first_model = next(gen, None)
        if first_model is None:
            raise RuntimeError(f"{ensemble_id}: no conformers were parsed!")

        # Per-residue numbering from the first model (determines array shapes).
        fa = _first_atom_per_residue(first_model)
        one_letter_first = normalize_resnames(first_model["resname"])[fa]
        resseq_first = first_model["resseq"][fa]
        tau_resseq, motif_spans = resolve_numbering(
            ensemble_id,
            resseq_first,
            one_letter_first,
            ec["tau_offset"],
            expected_motifs=_expected_motifs(ensemble_id),
        )
        n_res = len(tau_resseq)

        # consolidated arrays (allocated now that n_res is known)
        rg_mass = np.full(n_cap, np.nan, dtype=np.float64)
        rg_equal = np.full(n_cap, np.nan, dtype=np.float64)
        e2e = np.full(n_cap, np.nan, dtype=np.float64)
        res_sasa = np.full((n_cap, n_res), np.nan, dtype=np.float32)
        res_rsa = np.full((n_cap, n_res), np.nan, dtype=np.float32)
        apr_rsa = np.full((n_cap, 2), np.nan, dtype=np.float32)
        neighbor_counts = np.zeros((n_cap, n_res), dtype=np.int16)
        node_pos_ca = np.full((n_cap, n_res, 3), np.nan, dtype=np.float32)
        if save_contact_maps:
            contact_maps = np.zeros((n_cap, n_res, n_res), dtype=np.uint8)
        n_contacts = np.zeros(n_cap, dtype=np.int64)
        n_edges = np.zeros(n_cap, dtype=np.int64)
        mean_degree = np.full(n_cap, np.nan, dtype=np.float64)
        graph_density = np.full(n_cap, np.nan, dtype=np.float64)

        def handle(model: Dict, idx: int) -> None:
            nonlocal processed
            fa_m = _first_atom_per_residue(model)
            res_one_letter = normalize_resnames(model["resname"])[fa_m]
            if len(res_one_letter) != n_res:
                raise RuntimeError(
                    f"{ensemble_id} model {idx}: {len(res_one_letter)} residues "
                    f"!= {n_res} (models in this file must have identical lengths)"
                )
            # data-integrity audit: sequence content must match the first model
            # for EVERY conformer (the APR masks / tau numbering are derived
            # from model 0, so a same-length sequence difference in any later
            # model would silently misalign them). np.array_equal short-circuits
            # on the first mismatch, so checking every model is ~free.
            if not np.array_equal(res_one_letter, one_letter_first):
                print(f"[WARN] {ensemble_id} model {idx}: residue sequence differs "
                      f"from the first model; APR masks may be misaligned.")
            desc = compute_model_descriptors(
                model,
                res_one_letter,
                n_res,
                tau_resseq,
                motif_spans,
                graph_cfg=graph_cfg,
                probe=cfg.probe_radius,
                n_probe_points=cfg.n_probe_points,
                contact_cutoff=cfg.contact_cutoff,
                neighbor_cutoff=cfg.neighbor_cutoff,
            )
            g = desc["graph"]
            rg_mass[idx] = desc["rg_mass_weighted"]
            rg_equal[idx] = desc["rg_equal_weight"]
            e2e[idx] = desc["end_to_end"]
            res_sasa[idx] = desc["res_sasa"]
            res_rsa[idx] = desc["res_rsa"]
            apr_rsa[idx] = desc["apr_mean_rsa"]
            neighbor_counts[idx] = desc["neighbor_counts"]
            node_pos_ca[idx] = g["node_pos_ca"]
            if save_contact_maps:
                contact_maps[idx] = desc["contact_map"]
            n_contacts[idx] = desc["n_contacts"]
            n_edges[idx] = desc["n_edges"]
            mean_degree[idx] = desc["mean_degree"]
            graph_density[idx] = desc["graph_density"]

            if save_models:
                np.savez_compressed(
                    os.path.join(models_dir, f"model_{idx:05d}.npz"),
                    aa_idx=g["node_features"][:, :21].argmax(axis=1).astype(np.uint8),
                    node_features=g["node_features"],
                    node_pos_ca=g["node_pos_ca"],
                    node_pos_sc=g["node_pos_sc"],
                    edge_index=g["edge_index"],
                    edge_attr=g["edge_attr"],
                    tau_resseq=tau_resseq.astype(np.int32),
                    res_sasa=desc["res_sasa"],
                    res_rsa=desc["res_rsa"],
                    rg_mass_weighted=np.float32(desc["rg_mass_weighted"]),
                    rg_equal_weight=np.float32(desc["rg_equal_weight"]),
                    end_to_end=np.float32(desc["end_to_end"]),
                    apr_mean_rsa=desc["apr_mean_rsa"],
                )

            records.append({
                "model": idx,
                "conformer_id": conformer_ids.get(idx + 1 + cfg.start_at, ""),
                "rg_mass_weighted": float(desc["rg_mass_weighted"]),
                "rg_equal_weight": float(desc["rg_equal_weight"]),
                "end_to_end": float(desc["end_to_end"]),
                "apr1_mean_rsa": float(desc["apr_mean_rsa"][0]),
                "apr2_mean_rsa": float(desc["apr_mean_rsa"][1]),
                "n_edges": int(desc["n_edges"]),
                "mean_degree": float(desc["mean_degree"]),
                "graph_density": float(desc["graph_density"]),
                "n_contacts": int(desc["n_contacts"]),
            })
            processed += 1
            if processed % progress_every == 0:
                print(f"  [{ensemble_id}] processed {processed} conformers "
                      f"({time.time() - t0:.1f}s)")

        handle(first_model, 0)
        for idx, model in enumerate(gen, start=1):
            if idx >= n_cap:
                raise RuntimeError(
                    f"{ensemble_id}: found more than {n_cap} conformers (file may "
                    f"have been updated). Increase capacity or use --n-models."
                )
            handle(model, idx)

    if processed == 0:
        raise RuntimeError(f"{ensemble_id}: no conformers were parsed!")

    # ---------- consolidated output ----------------------------------------
    arrays = dict(
        rg_mass_weighted=rg_mass[:processed],
        rg_equal_weight=rg_equal[:processed],
        end_to_end=e2e[:processed],
        res_sasa=res_sasa[:processed],
        res_rsa=res_rsa[:processed],
        apr_mean_rsa=apr_rsa[:processed],
        neighbor_counts=neighbor_counts[:processed],
        node_pos_ca=node_pos_ca[:processed],
        n_contacts=n_contacts[:processed],
        n_edges=n_edges[:processed],
        mean_degree=mean_degree[:processed],
        graph_density=graph_density[:processed],
        tau_resseq=tau_resseq.astype(np.int32),
        aa_codes=np.asarray([str(a) for a in one_letter_first], dtype="U1"),
    )
    if save_contact_maps:
        arrays["contact_map"] = contact_maps[:processed]
    if conformer_ids:
        arrays["conformer_ids"] = np.asarray(
            [conformer_ids.get(i + 1 + cfg.start_at, "") for i in range(processed)],
            dtype="U32",
        )
    np.savez_compressed(os.path.join(out_dir, "ensemble_data.npz"), **arrays)

    # ---------- summary -----------------------------------------------------
    summary = {
        "ensemble_id": ensemble_id,
        "construct": ec["construct"],
        "method": ec["method"],
        "notes": ec["notes"],
        "n_models_expected": ec["n_expected_models"],
        "n_models_processed": processed,
        "n_residues": n_res,
        "tau_offset": ec["tau_offset"],
        "motif_spans_tau_numbering": {k: list(v) for k, v in motif_spans.items()},
        "processed_seconds": round(time.time() - t0, 1),
        "graph_config": graph_cfg.__dict__,
        "sasa": {"probe_radius": cfg.probe_radius, "n_probe_points": cfg.n_probe_points,
                 "vdw": "Chothia/NACCESS", "heavy_only": cfg.heavy_only},
        "statistics": {
            "rg_mass_weighted": {
                "mean": float(np.nanmean(rg_mass[:processed])),
                "std": float(np.nanstd(rg_mass[:processed])),
            },
            "rg_equal_weight": {
                "mean": float(np.nanmean(rg_equal[:processed])),
                "std": float(np.nanstd(rg_equal[:processed])),
            },
            "end_to_end": {
                "mean": float(np.nanmean(e2e[:processed])),
                "std": float(np.nanstd(e2e[:processed])),
            },
            "apr1_vqiink_mean_rsa": {
                "mean": float(np.nanmean(apr_rsa[:processed, 0])),
                "std": float(np.nanstd(apr_rsa[:processed, 0])),
            },
            "apr2_vqivyk_mean_rsa": {
                "mean": float(np.nanmean(apr_rsa[:processed, 1])),
                "std": float(np.nanstd(apr_rsa[:processed, 1])),
            },
            "n_edges": {"mean": float(np.nanmean(n_edges[:processed])),
                        "std": float(np.nanstd(n_edges[:processed]))},
            "mean_degree": {"mean": float(np.nanmean(mean_degree[:processed])),
                            "std": float(np.nanstd(mean_degree[:processed]))},
            "graph_density": {"mean": float(np.nanmean(graph_density[:processed])),
                              "std": float(np.nanstd(graph_density[:processed]))},
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import pandas as pd
        pd.DataFrame(records).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    except ImportError:
        pass

    print(f"[{ensemble_id}] done: {processed} conformers, {time.time() - t0:.1f}s")
    return summary
