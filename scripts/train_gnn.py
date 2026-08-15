"""Phase 4/5 - GNN training + ML evaluation entry point.

Runs the full node-level APR classification study:

  A. Main experiments     GCN (3 seeds) / GAT / GraphSAGE / MLP trained on
                          graph-level splits of PED00422 (full-length Tau);
                          test PR-AUC / ROC-AUC / F1 + confusion matrices +
                          training curves.
  B. Cross-ensemble       GCN trained on each ensemble, evaluated on all
                          three (PR-AUC transfer matrix).
  C. Ablations            GCN variants: no edge attributes (topology only),
                          no amino-acid identity, +rSA/SASA node features;
                          MLP = no-graph-structure baseline.
  D. Explainability       GNNExplainer edge masks on held-out conformers for
                          VQIINK/VQIVYK nodes.
  E. Embeddings           penultimate-layer node embeddings -> PCA + t-SNE
                          (colored by APR label and by source ensemble).
  F. Permutation importance for MLP node-feature groups.

All numeric results and figures are written under outputs/gnn/.
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.gnn import (  # noqa: E402
    MODEL_FACTORY,
    build_model,
    cross_ensemble_matrix,
    explain_apr_nodes,
    extract_embeddings,
    fit_scaler,
    load_ensemble_graphs,
    permutation_importance,
    predict_proba,
    run_split,
    stratified_graph_split,
    to_pyg_data,
)
from tau_mech.gnn import _PYG_IMPORT_ERROR as PYG_ERROR  # noqa: E402

OUT = os.path.join("outputs", "gnn")
ENSEMBLES = ["PED00422", "PED00192", "PED00443"]
# CPU thread cap for torch: for these tiny graphs (<=441 nodes) thread-launch
# overhead dominates, and a small cap keeps the SPH jobs (same machine)
# responsive. Override with TRAU_MECH_THREADS if desired.
torch.set_num_threads(int(os.environ.get("TRAU_MECH_THREADS", "2")))
# checkpoint for section A: section A is ~2.5 h of CPU; if a later section
# crashes, resume from the checkpoint instead of retraining everything.
CKPT = os.path.join(OUT, "main_checkpoint.pt")
MAIN_IN_DIM = 23  # PED00422 node features: 21 one-hot AA + hydropathy + pos


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print(f"[threads] torch intra-op threads = {torch.get_num_threads()}")
    if PYG_ERROR is not None:
        print(f"PyG import failed: {PYG_ERROR}")
        sys.exit(1)

    print("Loading graphs ...")
    data = {e: load_ensemble_graphs(e) for e in ENSEMBLES}
    for e, g in data.items():
        n_pos = sum(apr_pos_count(x) for x in g)
        print(f"  {e}: {len(g)} graphs, {n_pos} APR-positive node-labels")

    # ---------------- A. main experiments (train on PED00422) ---------------
    print("\n[A] Main experiments: train on PED00422 graph-level splits")
    tr, va, te = stratified_graph_split(data["PED00422"], seed=0)
    main_results = []
    if os.path.exists(CKPT):
        print("  [resume] loading section-A checkpoint ...")
        saved = torch.load(CKPT, weights_only=False)
        for c in saved:
            model = build_model(c["kind"], in_dim=MAIN_IN_DIM)
            model.load_state_dict(c["state_dict"])
            c["model"] = model
            main_results.append(c)
        print(f"  [resume] restored {len(main_results)} trained models "
              f"({CKPT})")
    else:
        for kind in ["GCN", "GAT", "GraphSAGE", "MLP"]:
            seeds = [0, 1, 2] if kind == "GCN" else [0]
            for sd in seeds:
                r = run_split(tr, va, te, kind, seed=sd, epochs=100, patience=20)
                print(f"  {kind:9s} seed={sd}  test PR-AUC={r['test']['pr_auc']:.4f} "
                      f"ROC={r['test']['roc_auc']:.4f} F1={r['test']['f1']:.4f} "
                      f"(best epoch {r['best_epoch']})")
                main_results.append({
                    "kind": kind, "seed": sd,
                    "best_epoch": r["best_epoch"],
                    "val_pr_auc": r["best_val"]["pr_auc"],
                    "test": r["test"],
                    "history": r["history"],
                    "model": r["model"],
                })
        # checkpoint so a crash in B-F does not waste the ~2.5 h of section A
        ckpt = []
        for m in main_results:
            c = {k: v for k, v in m.items() if k != "model"}
            c["state_dict"] = {k: v.clone()
                                for k, v in m["model"].state_dict().items()}
            ckpt.append(c)
        torch.save(ckpt, CKPT)
        print(f"  [ckpt] saved {len(ckpt)} trained models to {CKPT}")
    _plot_training_curves(main_results)
    _plot_pr_curves(main_results, data["PED00422"], te, tr)

    # ---------------- B. cross-ensemble transfer ----------------------------
    print("\n[B] Cross-ensemble GCN transfer")
    x = cross_ensemble_matrix(ENSEMBLES, "GCN", seed=0, epochs=100, patience=20)
    _plot_transfer_matrix(x)

    # ---------------- C. ablations ------------------------------------------
    print("\n[C] Ablations (GCN on PED00422 split)")
    ablations = {
        "full": {},
        "no_edge_attr": {"use_edge_attr": False},
        "no_aa": {"no_aa": True},
        "rsa_augment": {"rsa_augment": True},
    }
    ab_results = {}
    for name, kw in ablations.items():
        r = run_split(tr, va, te, "GCN", seed=0, epochs=100, patience=20, **kw)
        ab_results[name] = r["test"]
        print(f"  {name:14s} test PR-AUC={r['test']['pr_auc']:.4f} "
              f"ROC={r['test']['roc_auc']:.4f}")
    mlp_r = [m for m in main_results if m["kind"] == "MLP"][0]
    ab_results["mlp_no_graph"] = mlp_r["test"]
    print(f"  {'mlp_no_graph':14s} test PR-AUC={mlp_r['test']['pr_auc']:.4f}")

    # ---------------- D. explainability -------------------------------------
    print("\n[D] GNNExplainer on held-out PED00422 conformers")
    gcn0 = next(m for m in main_results if m["kind"] == "GCN" and m["seed"] == 0)
    explain = {"graphs": []}
    for gi in range(min(2, len(te))):
        d = to_pyg_data([te[gi]])[0]
        ex = explain_apr_nodes(gcn0["model"], d)
        explain["graphs"].append({"conformer": te[gi]["conformer_id"], "nodes": ex})
        print(f"  conformer {te[gi]['conformer_id']}: explained {len(ex)} APR nodes")

    # ---------------- E. embeddings -----------------------------------------
    print("\n[E] Node embeddings -> PCA + t-SNE")
    emb_all = {}
    for e in ENSEMBLES:
        emb = extract_embeddings(gcn0["model"], data[e], max_nodes=60000)
        emb_all[e] = emb
    _plot_embeddings(emb_all)

    # ---------------- F. permutation importance ------------------------------
    print("\n[F] Permutation importance (MLP)")
    mlp_model = mlp_r["model"]
    rng = np.random.default_rng(0)
    perm = permutation_importance(mlp_model, te, rng)
    for name, v in perm.items():
        print(f"  {name:12s} PR-AUC drop {v['drop_mean']:.4f} +/- {v['drop_std']:.4f} "
              f"(base {v['base_pr_auc']:.4f})")

    # ---------------- summary ------------------------------------------------
    summary = {
        "main": [{"kind": m["kind"], "seed": m["seed"],
                  "val_pr_auc": m["val_pr_auc"], "test": m["test"]}
                 for m in main_results],
        "cross_ensemble": x["matrix_rows"],
        "ablations": ab_results,
        "permutation_importance": perm,
        "explainability": explain,   # GNNExplainer masks, saved for audit
        "n_graphs": {e: len(data[e]) for e in ENSEMBLES},
        "wall_time_s": time.time() - t0,
        "note": ("Node-level APR classification (VQIINK 275-280, VQIVYK "
                 "306-311). Graph-level splits, PR-AUC primary. See README "
                 "phases 4-5 for the full protocol."),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {OUT}/")


def apr_pos_count(g) -> int:
    from tau_mech.gnn import apr_labels
    return int(apr_labels(g["tau_resseq"]).sum())


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plot_training_curves(main_results) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in main_results:
        h = m["history"]
        axes[0].plot(h["train_loss"], lw=0.8,
                     label=f"{m['kind']} s{m['seed']}")
        axes[1].plot(h["val_pr_auc"], lw=0.8,
                     label=f"{m['kind']} s{m['seed']}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val PR-AUC")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=150)
    plt.close(fig)


def _plot_pr_curves(main_results, test_graphs, te, tr) -> None:
    """PR + ROC curves on the PED00422 test split (best seed per kind)."""
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    scaler = fit_scaler(tr)
    test_d = to_pyg_data(te, scaler)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in main_results:
        if m["seed"] != 0:
            continue
        y, p = predict_proba(m["model"], test_d)
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(rec, prec, lw=1.2,
                     label=f"{m['kind']} (AP={average_precision_score(y, p):.3f})")
        axes[1].plot(fpr, tpr, lw=1.2,
                     label=f"{m['kind']} (AUC={roc_auc_score(y, p):.3f})")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall (PED00422 test)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC (PED00422 test)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pr_roc_curves.png"), dpi=150)
    plt.close(fig)


def _plot_transfer_matrix(x) -> None:
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
    ax.set_title("PR-AUC transfer matrix (GCN)")
    for i in range(len(srcs)):
        for j in range(len(tgts)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.55 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_matrix.png"), dpi=150)
    plt.close(fig)
    np.save(os.path.join(OUT, "transfer_matrix.npy"), M)


def _plot_embeddings(emb_all) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    # PCA (fast, all nodes)
    pca = PCA(n_components=2, random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key in zip(axes, ["y", "src"]):
        pass
    z = []
    ys = []
    srcs = []
    taus = []
    for e, d in emb_all.items():
        z.append(d["emb"]); ys.append(d["y"]); taus.append(d["tau_resseq"])
        srcs.append(np.full(len(d["y"]), ENSEMBLES.index(e)))
    Z = np.concatenate(z); Y = np.concatenate(ys)
    S = np.concatenate(srcs); T = np.concatenate(taus)

    zp = pca.fit_transform(Z)
    sc = axes[0].scatter(zp[:, 0], zp[:, 1], c=Y, s=4, cmap="coolwarm",
                         vmin=0, vmax=1)
    axes[0].set_title(f"PCA (n={len(Y)}) colored by APR label")
    fig.colorbar(sc, ax=axes[0])
    cmap = plt.get_cmap("tab10")
    for si, e in enumerate(ENSEMBLES):
        m = S == si
        axes[1].scatter(zp[m, 0], zp[m, 1], s=4, color=cmap(si), alpha=0.5,
                        label=e)
    axes[1].set_title("PCA colored by source ensemble")
    axes[1].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "embeddings_pca.png"), dpi=150)
    plt.close(fig)

    # t-SNE on a balanced subsample (~4000 nodes)
    rng = np.random.default_rng(0)
    pos = np.where(Y == 1)[0]
    neg = np.where(Y == 0)[0]
    k = min(len(pos), 4000)
    idx = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False),
                          rng.choice(neg, min(4000 - k, len(neg)), replace=False)])
    ts = TSNE(n_components=2, perplexity=30, random_state=0, n_jobs=1)
    zt = ts.fit_transform(Z[idx])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(zt[:, 0], zt[:, 1], c=Y[idx], s=5, cmap="coolwarm",
                         vmin=0, vmax=1)
    axes[0].set_title(f"t-SNE (n={len(idx)}) colored by APR label")
    fig.colorbar(sc, ax=axes[0])
    for si, e in enumerate(ENSEMBLES):
        m = S[idx] == si
        axes[1].scatter(zt[m, 0], zt[m, 1], s=5, color=cmap(si), alpha=0.5,
                        label=e)
    axes[1].set_title("t-SNE colored by source ensemble")
    axes[1].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "embeddings_tsne.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
