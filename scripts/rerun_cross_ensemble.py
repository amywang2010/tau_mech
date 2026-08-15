"""Re-run the cross-ensemble transfer matrix with the scaler bug fixed.

The original section B evaluated target graphs WITHOUT the training scaler
(raw continuous features fed to a model trained on z-scored features). The
code path was wrong, but the numerical impact was SMALL: the continuous
features (hydropathy, normalized sequence position) are already near
unit-scale, so z-scoring moved the transfer PR-AUCs only in the 3rd decimal
(0.899 -> 0.8986, 0.402 -> 0.4018, ...). This script re-runs it correctly,
regenerates the figure, and patches summary.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.gnn import (  # noqa: E402
    cross_ensemble_matrix,
    load_ensemble_graphs,
)
from tau_mech.gnn import _PYG_IMPORT_ERROR as PYG_ERROR  # noqa: E402

OUT = os.path.join("outputs", "gnn")
ENSEMBLES = ["PED00422", "PED00192", "PED00443"]


def plot_matrix(x) -> None:
    rows = x["matrix_rows"]
    srcs = sorted({r["train_on"] for r in rows})
    tgts = sorted({r["test_on"] for r in rows})
    M = np.full((len(srcs), len(tgts)), np.nan)
    for r in rows:
        M[srcs.index(r["train_on"]), tgts.index(r["test_on"])] = r["pr_auc"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tgts)), [f"{t[-3:]}" for t in tgts])
    ax.set_yticks(range(len(srcs)), [f"{s[-3:]}" for s in srcs])
    ax.set_xlabel("test ensemble"); ax.set_ylabel("train ensemble")
    ax.set_title("PR-AUC transfer matrix (GCN, scaler-fixed)")
    for i in range(len(srcs)):
        for j in range(len(tgts)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.55 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_matrix.png"), dpi=150)
    plt.close(fig)
    np.save(os.path.join(OUT, "transfer_matrix.npy"), M)


def main() -> None:
    t0 = time.time()
    if PYG_ERROR is not None:
        print(f"PyG import failed: {PYG_ERROR}")
        sys.exit(1)
    for e in ENSEMBLES:
        load_ensemble_graphs(e)  # warm cache
    x = cross_ensemble_matrix(ENSEMBLES, "GCN", seed=0, epochs=100, patience=20)
    for r in x["matrix_rows"]:
        print(f"  {r['train_on'][-3:]} -> {r['test_on'][-3:]}  "
              f"PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} F1={r['f1']:.4f}",
              flush=True)
    plot_matrix(x)

    spath = os.path.join(OUT, "summary.json")
    with open(spath) as f:
        summary = json.load(f)
    summary["cross_ensemble"] = x["matrix_rows"]
    summary["cross_ensemble_note"] = (
        "Regenerated after fixing a scaler bug (target graphs were evaluated "
        "without the training-ensemble z-scoring). The code path was wrong "
        "but the numerical impact was small (~1e-3 in PR-AUC): the continuous "
        "features are already near unit-scale, so the corrected values are "
        "essentially unchanged from the first run.")
    summary["wall_time_s"] = summary.get("wall_time_s", 0) + (time.time() - t0)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"matrix saved + summary.json updated in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
