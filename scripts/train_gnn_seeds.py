"""GNN seed-variance polish (2026-09-04, rescoped plan, CPU-cheap).

Purpose: the main study ran GCN with 3 seeds (0/1/2) but GAT/GraphSAGE with
seed 0 only. A reviewer can ask whether the model-comparison ordering is
seed-robust. This script extends the seed set on the IDENTICAL PED00422
graph-level split (stratified_graph_split, seed=0 -> same split object),
training:

  * GAT        seeds 1, 2   (new; joins seed 0 from the main run)
  * GraphSAGE  seeds 1, 2   (new; joins seed 0 from the main run)
  * GCN        seeds 3, 4   (extends the existing 0/1/2 set to n=5)

Protocol: epochs=100, patience=20, as-delivered features - identical to the
main run's section A. The reference seed-0/1/2 results are read from
outputs/gnn/gnn_summary.json (written by the main run); if that file is
absent the script still writes its own runs and records the absence.

Output: outputs/gnn/seed_polish.json with per-run test metrics and per-kind
aggregates (mean/sd/min/max over ALL seeds incl. the main run's).

Run:  python scripts/train_gnn_seeds.py
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
    _PYG_IMPORT_ERROR as PYG_ERROR,
    load_ensemble_graphs,
    run_split,
    stratified_graph_split,
)

OUT_JSON = os.path.join("outputs", "gnn", "seed_polish.json")
SUMMARY = os.path.join("outputs", "gnn", "summary.json")  # main-run record (defect fix 2026-09-04: was gnn_summary.json, which does not exist - the reference merge silently no-oped)

torch.set_num_threads(2)

PLAN = {
    "GAT": [1, 2],
    "GraphSAGE": [1, 2],
    "GCN": [3, 4],
}


def load_main_reference():
    """Main-run reference results for the PLAN kinds (or empty + status)."""
    if not os.path.exists(SUMMARY):
        return [], "summary.json not found"
    try:
        s = json.load(open(SUMMARY))
        ref = [m for m in s.get("main", [])
               if m.get("kind") in PLAN and "test" in m]
        return ref, f"{len(ref)} reference runs loaded"
    except Exception as e:  # noqa: BLE001 - record, never crash the record
        return [], f"summary parse failed: {e!r}"


def aggregate(runs, main_ref):
    """Per-kind aggregates over ALL seeds (new runs + main reference)."""
    agg = {}
    for kind in PLAN:
        all_runs = [{"seed": r["seed"], "pr_auc": r["test"]["pr_auc"],
                     "roc_auc": r["test"]["roc_auc"], "f1": r["test"]["f1"]}
                    for r in runs if r["kind"] == kind]
        all_runs += [{"seed": m["seed"], "pr_auc": m["test"]["pr_auc"],
                      "roc_auc": m["test"]["roc_auc"], "f1": m["test"]["f1"]}
                     for m in main_ref if m["kind"] == kind]
        pr = np.array([r["pr_auc"] for r in all_runs])
        roc = np.array([r["roc_auc"] for r in all_runs])
        agg[kind] = {
            "seeds": sorted(r["seed"] for r in all_runs),
            "pr_auc_mean": float(pr.mean()), "pr_auc_sd": float(pr.std(ddof=1)) if len(pr) > 1 else 0.0,
            "pr_auc_min": float(pr.min()), "pr_auc_max": float(pr.max()),
            "roc_auc_mean": float(roc.mean()),
            "n_runs": len(all_runs),
        }
    return agg


def reagregate_mode() -> None:
    """Rebuild aggregates from the existing record through the SAME
    aggregation code (no re-training; training is deterministic per seed
    and its raw results are untouched). The defective pre-fix record is
    preserved as seed_polish_pre_merge_fix.json and the amendment is
    documented inside the record."""
    rec = json.load(open(OUT_JSON))
    backup = OUT_JSON.replace(".json", "_pre_merge_fix.json")
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            json.dump(rec, f, indent=2)
    main_ref, main_status = load_main_reference()
    runs = rec["new_runs"]
    agg = aggregate(runs, main_ref)
    for kind, a in agg.items():
        print(f"  [{kind}] n={a['n_runs']} seeds={a['seeds']} "
              f"PR-AUC {a['pr_auc_mean']:.4f} +/- {a['pr_auc_sd']:.4f}",
              flush=True)
    rec["main_summary_status"] = main_status
    rec["aggregates_all_seeds"] = agg
    rec["amendment"] = (
        "2026-09-04: the original run recorded main_summary_status = "
        "'gnn_summary.json not found' - a wrong reference filename in "
        "this script, so aggregates covered only the 6 new runs (n=2 "
        "per kind). Aggregates rebuilt here through the identical "
        "aggregation code with the corrected reference (summary.json); "
        "new_runs are untouched; the pre-fix record is preserved as "
        "seed_polish_pre_merge_fix.json.")
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"reaggregated -> {OUT_JSON}", flush=True)


def main() -> None:
    if "--reaggregate" in sys.argv:
        reagregate_mode()
        return
    t0 = time.time()
    if PYG_ERROR is not None:
        print(f"PyG import failed: {PYG_ERROR}")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print("Loading PED00422 graphs ...", flush=True)
    graphs = load_ensemble_graphs("PED00422")
    tr, va, te = stratified_graph_split(graphs, seed=0)
    print(f"  split: {len(tr)}/{len(va)}/{len(te)} graphs", flush=True)

    runs = []
    for kind, seeds in PLAN.items():
        for sd in seeds:
            r = run_split(tr, va, te, kind, seed=sd, epochs=100, patience=20)
            print(f"  {kind:9s} seed={sd}  test PR-AUC={r['test']['pr_auc']:.4f} "
                  f"ROC={r['test']['roc_auc']:.4f} F1={r['test']['f1']:.4f} "
                  f"(best epoch {r['best_epoch']})", flush=True)
            runs.append({"kind": kind, "seed": sd,
                         "best_epoch": r["best_epoch"],
                         "val_pr_auc": r["best_val"]["pr_auc"],
                         "test": r["test"]})

    # Merge with the main run's seeds for the aggregates (shared code path
    # with --reaggregate: one aggregation implementation, one truth).
    main_ref, main_status = load_main_reference()
    agg = aggregate(runs, main_ref)
    for kind, a in agg.items():
        print(f"  [{kind}] n={a['n_runs']} seeds={a['seeds']} "
              f"PR-AUC {a['pr_auc_mean']:.4f} +/- {a['pr_auc_sd']:.4f}",
              flush=True)

    record = {
        "purpose": "seed-variance polish: GAT/GraphSAGE n=3, GCN n=5 on the "
                   "identical stratified seed-0 split; model-comparison "
                   "robustness evidence",
        "protocol": "epochs=100 patience=20, as-delivered features, identical "
                    "to train_gnn.py section A",
        "main_summary_status": main_status,
        "new_runs": runs,
        "aggregates_all_seeds": agg,
        "wall_clock_s": time.time() - t0,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(record, f, indent=2)
    print(f"saved -> {OUT_JSON}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
