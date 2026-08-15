"""Supplement experiments for the GNN study (corrected ablations).

The main run's ``no_edge_attr`` ablation was vacuous: the GCN/GAT/GraphSAGE
forwards do not consume edge attributes, so toggling them changed nothing.
This script runs the MEANINGFUL structural ablation and the edge-aware model:

  1. ``no_spatial_edges`` - GCN trained on sequential backbone edges only
     (3D contact topology removed). Any loss of PR-AUC measures the
     information carried by 3D contacts - the paper's mechanistic crux.
  2. ``GAT_edge`` - GAT that consumes the edge attributes
     [distance, seq_sep, is_spatial] via GATConv(edge_dim=3).

Both use the identical PED00422 graph-level split (seed 0) as the main run,
so they are directly comparable with the main-run GCN (PR-AUC 0.9030) and
MLP (0.7929) baselines.

Outputs: outputs/gnn/supplement.json + outputs/gnn/supplement_ablations.png
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.gnn import (  # noqa: E402
    load_ensemble_graphs,
    run_split,
    stratified_graph_split,
)
from tau_mech.gnn import _PYG_IMPORT_ERROR as PYG_ERROR  # noqa: E402

OUT = os.path.join("outputs", "gnn")


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if PYG_ERROR is not None:
        print(f"PyG import failed: {PYG_ERROR}")
        sys.exit(1)

    graphs = load_ensemble_graphs("PED00422")
    tr, va, te = stratified_graph_split(graphs, seed=0)

    results = {}

    # ---- 1. corrected structural ablation: sequential edges only -----------
    print("[1] GCN, sequential backbone edges only (no spatial contacts)")
    r = run_split(tr, va, te, "GCN", seed=0, epochs=100, patience=20,
                  keep_spatial=False)
    results["gcn_no_spatial_edges"] = {
        "test": {k: v for k, v in r["test"].items() if k != "confusion_matrix"},
        "best_epoch": r["best_epoch"],
    }
    print(f"    test PR-AUC={r['test']['pr_auc']:.4f} ROC={r['test']['roc_auc']:.4f} "
          f"F1={r['test']['f1']:.4f} (best epoch {r['best_epoch']})")

    # ---- 2. edge-attribute-aware GAT ---------------------------------------
    print("[2] GAT with edge attributes (distance, seq_sep, is_spatial)")
    r2 = run_split(tr, va, te, "GAT", seed=0, epochs=100, patience=20,
                   model_kwargs={"use_edge_attr": True})
    results["gat_edge_attr"] = {
        "test": {k: v for k, v in r2["test"].items() if k != "confusion_matrix"},
        "best_epoch": r2["best_epoch"],
    }
    print(f"    test PR-AUC={r2['test']['pr_auc']:.4f} ROC={r2['test']['roc_auc']:.4f} "
          f"F1={r2['test']['f1']:.4f} (best epoch {r2['best_epoch']})")

    # ---- comparison vs the main-run baselines ------------------------------
    main_vals = {
        "GCN (full)": 0.9030, "MLP (no graph)": 0.7929,
        "GraphSAGE": 0.9997, "GAT (topology)": 0.9366,
    }
    sup_vals = {
        "GCN no-spatial": r["test"]["pr_auc"],
        "GAT + edge attrs": r2["test"]["pr_auc"],
    }
    results["comparison_pr_auc"] = {**main_vals, **sup_vals}
    results["wall_time_s"] = time.time() - t0
    results["note"] = (
        "Supplement to the main GNN run. 'no_edge_attr' in the main run was "
        "vacuous (the GNN forwards never consume edge_attr); the meaningful "
        "structural ablation is no_spatial_edges. GAT_edge additionally tests "
        "whether edge distance information changes APR classification.")

    with open(os.path.join(OUT, "supplement.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure --------------------------------------------------------------
    names = list(main_vals) + list(sup_vals)
    vals = [main_vals[k] for k in main_vals] + [sup_vals[k] for k in sup_vals]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(names)), vals, color="0.6")
    bars[-2].set_color("steelblue")   # no-spatial ablation
    bars[-1].set_color("crimson")     # edge-aware GAT
    ax.axhline(0.7929, ls="--", color="0.4", lw=0.8)
    ax.text(len(names) - 0.5, 0.80, "MLP baseline", ha="right", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test PR-AUC")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("APR classification: structural ablation + edge-aware GAT")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "supplement_ablations.png"), dpi=150)
    plt.close(fig)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {OUT}/supplement.json")


if __name__ == "__main__":
    main()
