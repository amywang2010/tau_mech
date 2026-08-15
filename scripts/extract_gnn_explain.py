"""Reconstruct and persist the GNNExplainer section-D results.

The first GNN run computed the GNNExplainer edge masks but did not save them
to summary.json. This script rebuilds them from the section-A checkpoint
(GCN seed 0) on the same held-out conformers (same graph-level split, seed 0)
and writes outputs/gnn/explainability.json, then merges it into summary.json.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.gnn import (  # noqa: E402
    build_model,
    explain_apr_nodes,
    load_ensemble_graphs,
    stratified_graph_split,
    to_pyg_data,
)

OUT = os.path.join("outputs", "gnn")
CKPT = os.path.join(OUT, "main_checkpoint.pt")


def main() -> None:
    if not os.path.exists(CKPT):
        print("checkpoint not found - nothing to do")
        sys.exit(1)
    saved = torch.load(CKPT, weights_only=False)
    gcn0 = next(c for c in saved if c["kind"] == "GCN" and c["seed"] == 0)
    model = build_model("GCN", in_dim=23)
    model.load_state_dict(gcn0["state_dict"])
    model.eval()

    graphs = load_ensemble_graphs("PED00422")
    _, _, te = stratified_graph_split(graphs, seed=0)

    explain = {"graphs": []}
    for gi in range(min(2, len(te))):
        d = to_pyg_data([te[gi]])[0]
        ex = explain_apr_nodes(model, d)
        explain["graphs"].append({"conformer": te[gi]["conformer_id"],
                                  "nodes": ex})
        print(f"  conformer {te[gi]['conformer_id']}: "
              f"{len(ex)} APR nodes explained", flush=True)

    with open(os.path.join(OUT, "explainability.json"), "w") as f:
        json.dump(explain, f, indent=2)

    # merge into summary.json (preserve existing content)
    spath = os.path.join(OUT, "summary.json")
    with open(spath) as f:
        summary = json.load(f)
    summary["explainability"] = explain
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {OUT}/explainability.json (+ merged into summary.json)")


if __name__ == "__main__":
    main()
