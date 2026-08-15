"""Re-run the rsa_augment ablation with the corrected SASA (self-occlusion fixed).

The original ablation used per-model res_rsa/res_sasa computed BEFORE the SASA
self-occlusion bug was fixed (2026-08-14), which under-estimated SASA for
partially-buried residues by up to ~2x. This script re-runs just that one
ablation (full / rsa_augment) on the fixed npz and patches summary.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.gnn import (  # noqa: E402
    load_ensemble_graphs,
    run_split,
    stratified_graph_split,
    to_pyg_data,
)
from tau_mech.gnn import _PYG_IMPORT_ERROR as PYG_ERROR  # noqa: E402

OUT = os.path.join("outputs", "gnn")
torch.set_num_threads(int(os.environ.get("TRAU_MECH_THREADS", "2")))


def main() -> None:
    t0 = time.time()
    if PYG_ERROR is not None:
        print(f"PyG import failed: {PYG_ERROR}")
        sys.exit(1)
    graphs = load_ensemble_graphs("PED00422")
    tr, va, te = stratified_graph_split(graphs, seed=0)
    print(f"PED00422: {len(tr)} train / {len(va)} val / {len(te)} test graphs")

    results = {}
    for name, kw in [("full", {}), ("rsa_augment", {"rsa_augment": True})]:
        r = run_split(tr, va, te, "GCN", seed=0, epochs=100, patience=20, **kw)
        results[name] = r["test"]
        print(f"  {name:14s} test PR-AUC={r['test']['pr_auc']:.4f} "
              f"ROC={r['test']['roc_auc']:.4f} F1={r['test']['f1']:.4f}")

    spath = os.path.join(OUT, "summary.json")
    with open(spath) as f:
        summary = json.load(f)
    ablations = summary.get("ablations", {})
    for name, res in results.items():
        ablations[name] = res
    ablations["rsa_augment_note"] = (
        "Regenerated with the corrected SASA (self-occlusion bug fixed "
        "2026-08-14); supersedes the first run's rsa_augment row, which used "
        "rASA values under-estimated by up to ~2x for buried residues.")
    summary["ablations"] = ablations
    summary["wall_time_s"] = summary.get("wall_time_s", 0) + (time.time() - t0)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary.json updated in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
