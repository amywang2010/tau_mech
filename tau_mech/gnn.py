"""Phase 4/5 - geometric GNNs for per-residue aggregation-prone-region (APR)
classification on the Tau conformational ensemble graphs.

Task
----
Given the residue-level geometric graph of one Tau conformation (built in
Phase 1), classify every residue as belonging (1) or not (0) to an
aggregation-prone region. The two APR hexapeptides of Tau-441 are
VQIINK (PHF6*, Tau 275-280) and VQIVYK (PHF6, Tau 306-311); labels are
assigned from the Tau-441 numbering carried in every graph (``tau_resseq``),
so the same labeling rule applies uniformly to the full-length and K18
constructs.

Why this task
-------------
The central mechanobiology hypothesis of this study is that mechanical shear
may alter the *accessibility* of aggregation-prone regions (the
conformational-susceptibility question). The GNN trained here is the learned
"APR detector": if APR membership is recoverable from sequence features plus
3D contact topology, the model becomes a probe we can apply to
shear-perturbed conformations (Phase 3 SPH output) to ask whether the
geometric context of the APRs changes under shear. The cross-ensemble
transfer experiments additionally test whether the *generated* idpGAN
ensemble (PED00443) reproduces the APR-relevant geometry of the
experimentally constrained K18 ensemble (PED00192).

Models
------
GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2018), GraphSAGE
(Hamilton et al., 2017) and a structure-blind MLP baseline (no message
passing). Node-level binary classification with class-weighted BCE; PR-AUC
is the primary metric because the positive class is heavily imbalanced
(~2.7% of residues).

Protocol (leakage-safe)
-----------------------
* Splits are made at the GRAPH level (whole conformers), never at the node
  level, so residues of one conformer never straddle train and test.
* Continuous feature columns are z-scored with statistics fit on the
  training split only.
* Early stopping on validation PR-AUC; the operating threshold is chosen on
  the validation PR curve (max F1) and applied to the test split.
* Headline results are averaged over 3 seeds (std reported).
* Cross-ensemble transfer is evaluated in both directions.

Explainability & interpretation
-------------------------------
* GNNExplainer edge masks on held-out conformers for APR nodes.
* Permutation feature importance on the MLP baseline (node features).
* Node-embedding visualization (PCA + t-SNE) across ensembles.
* Ablations: no edge attributes (topology only), no amino-acid identity,
  structure-derived accessibility appended (rSA/SASA), MLP (no structure).
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import AGGREGATION_PRONE_MOTIFS

try:
    from torch_geometric.data import Data
    try:  # PyG >= 2.5: loader module; older: torch_geometric.data.DataLoader
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch_geometric.data import DataLoader
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except Exception as e:  # pragma: no cover - import guard for environments
    Data = None
    DataLoader = None
    GATConv = GCNConv = SAGEConv = None
    _PYG_IMPORT_ERROR = e
else:
    _PYG_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Data loading and labeling
# ---------------------------------------------------------------------------


def apr_labels(tau_resseq: np.ndarray) -> np.ndarray:
    """Binary APR labels from Tau-441 numbering (inclusive spans)."""
    y = np.zeros(len(tau_resseq), dtype=np.float32)
    for lo, hi in AGGREGATION_PRONE_MOTIFS.values():
        y[(tau_resseq >= lo) & (tau_resseq <= hi)] = 1.0
    return y


def load_ensemble_graphs(ensemble_id: str,
                         out_dir: str = "outputs") -> List[Dict]:
    """Load all per-model graphs of one ensemble from the Phase-1 outputs."""
    d = os.path.join(out_dir, ensemble_id, "models")
    paths = sorted(glob.glob(os.path.join(d, "model_*.npz")))
    graphs = []
    for p in paths:
        g = np.load(p)
        graphs.append({
            "node_features": g["node_features"].astype(np.float32),
            "edge_index": g["edge_index"].astype(np.int64),
            "edge_attr": g["edge_attr"].astype(np.float32),
            "tau_resseq": g["tau_resseq"].astype(np.int64),
            "res_rsa": g["res_rsa"].astype(np.float32),
            "res_sasa": g["res_sasa"].astype(np.float32),
            "conformer_id": os.path.basename(p),
        })
    return graphs


CONTINUOUS_COLS = [21, 22]          # hydropathy, normalized position
RSA_COLS = [23, 24]                 # rSA, SASA (only in the rsa_augment mode)


def _feature_columns(rsa_augment: bool) -> List[int]:
    return CONTINUOUS_COLS + (RSA_COLS if rsa_augment else [])


def fit_scaler(graphs: Sequence[Dict], rsa_augment: bool = False) -> Dict:
    """Z-score statistics for the continuous columns, fit on training graphs."""
    cols = _feature_columns(rsa_augment)
    xs = []
    for g in graphs:
        x = g["node_features"]
        if rsa_augment:
            x = np.concatenate([x, g["res_rsa"][:, None], g["res_sasa"][:, None]],
                               axis=1)
        xs.append(x)
    allx = np.concatenate(xs, axis=0)
    mean = allx[:, cols].mean(axis=0)
    std = allx[:, cols].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"cols": cols, "mean": mean, "std": std, "rsa_augment": rsa_augment}


def _augment_features(g: Dict, rsa_augment: bool) -> np.ndarray:
    x = g["node_features"]
    if rsa_augment:
        x = np.concatenate([x, g["res_rsa"][:, None], g["res_sasa"][:, None]],
                           axis=1)
    return x


def to_pyg_data(graphs: Sequence[Dict], scaler: Optional[Dict] = None,
                rsa_augment: bool = False,
                use_edge_attr: bool = True,
                keep_spatial: bool = True) -> List["Data"]:
    """Convert graph dicts into torch_geometric Data objects.

    Edge attributes: [min heavy-atom distance (NaN-filled to 0 for pure
    sequential edges), sequence separation, is_spatial flag]. With
    ``keep_spatial=False`` (the structural ablation) only the sequential
    backbone edges (NaN distance) are kept - the graph then encodes chain
    connectivity alone, and any performance loss measures the information
    carried by 3D contact topology.
    """
    datas = []
    for g in graphs:
        x = _augment_features(g, rsa_augment).copy()
        if scaler is not None:
            cols = scaler["cols"]
            x[:, cols] = (x[:, cols] - scaler["mean"]) / scaler["std"]
        ea = g["edge_attr"]  # (E, 2): [distance (may be NaN), seq_sep]
        ei = g["edge_index"]
        if not keep_spatial:
            keep = np.isnan(ea[:, 0])  # sequential edges carry NaN distance
            ea = ea[keep]
            ei = ei[:, keep]
        if use_edge_attr:
            dist = np.nan_to_num(ea[:, 0], nan=0.0)
            is_spatial = (~np.isnan(ea[:, 0])).astype(np.float32)
            edge_attr = np.stack([dist, ea[:, 1], is_spatial], axis=1).astype(np.float32)
        else:
            edge_attr = np.zeros((ea.shape[0], 3), dtype=np.float32)
        datas.append(Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(ei),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(apr_labels(g["tau_resseq"])),
            tau_resseq=torch.from_numpy(g["tau_resseq"]),
        ))
    return datas


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = GCNConv(hidden, out_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.act = torch.relu

    def forward(self, x, edge_index, edge_attr=None, return_emb: bool = False):
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        emb = x
        x = self.drop(x)
        out = self.head(x, edge_index).squeeze(-1)
        return (out, emb) if return_emb else out


class GAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1,
                 dropout: float = 0.3, heads: int = 4,
                 use_edge_attr: bool = False):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        kw = {"edge_dim": 3} if use_edge_attr else {}
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads,
                             dropout=dropout, **kw)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads,
                             dropout=dropout, **kw)
        self.head = GATConv(hidden, out_dim, heads=1, **kw)
        self.drop = torch.nn.Dropout(dropout)
        self.act = torch.relu

    def forward(self, x, edge_index, edge_attr=None, return_emb: bool = False):
        ea = edge_attr if self.use_edge_attr else None
        x = self.act(self.conv1(x, edge_index, edge_attr=ea))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index, edge_attr=ea))
        emb = x
        x = self.drop(x)
        out = self.head(x, edge_index, edge_attr=ea).squeeze(-1)
        return (out, emb) if return_emb else out


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.head = SAGEConv(hidden, out_dim, aggr="mean")
        self.drop = torch.nn.Dropout(dropout)
        self.act = torch.relu

    def forward(self, x, edge_index, edge_attr=None, return_emb: bool = False):
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        emb = x
        x = self.drop(x)
        out = self.head(x, edge_index).squeeze(-1)
        return (out, emb) if return_emb else out


class MLP(torch.nn.Module):
    """Structure-blind baseline: node-level MLP over node features only."""

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index=None, edge_attr=None, return_emb: bool = False):
        h = self.net[:4](x)
        emb = h
        out = self.net[4:](h).squeeze(-1)
        return (out, emb) if return_emb else out


MODEL_FACTORY = {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "MLP": MLP}


def build_model(kind: str, in_dim: int, hidden: int = 64, **kw) -> torch.nn.Module:
    if kind not in MODEL_FACTORY:
        raise ValueError(f"unknown model kind {kind!r}")
    return MODEL_FACTORY[kind](in_dim, hidden=hidden, **kw)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_proba(model: torch.nn.Module, datas: List[Data],
                  batch: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in DataLoader(datas, batch_size=batch):
            out = model(data.x, data.edge_index, data.edge_attr)
            ps.append(torch.sigmoid(out).numpy())
            ys.append(data.y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate(y: np.ndarray, p: np.ndarray,
             threshold: Optional[float] = None) -> Dict:
    """PR-AUC (primary), ROC-AUC, and thresholded P/R/F1 + confusion matrix."""
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_score, recall_score,
                                 roc_auc_score)
    ap = float(average_precision_score(y, p))
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = float("nan")
    if threshold is None:
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y, p)
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        threshold = float(thr[np.argmax(f1s)])
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat)
    return {
        "pr_auc": ap,
        "roc_auc": roc,
        "threshold": threshold,
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(2 * precision_score(y, yhat, zero_division=0)
                    * recall_score(y, yhat, zero_division=0)
                    / (precision_score(y, yhat, zero_division=0)
                       + recall_score(y, yhat, zero_division=0) + 1e-12)),
        "confusion_matrix": cm.tolist(),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }


def train_model(train_g: Sequence[Dict], val_g: Sequence[Dict],
                model_kind: str, seed: int = 0, hidden: int = 64,
                epochs: int = 100, patience: int = 20, batch: int = 32,
                lr: float = 1e-3, weight_decay: float = 5e-4,
                rsa_augment: bool = False,
                use_edge_attr: bool = True,
                no_aa: bool = False,
                keep_spatial: bool = True,
                model_kwargs: Optional[Dict] = None) -> Dict:
    """Train one model; returns the model + full history + best val metrics.

    ``no_aa`` drops the one-hot amino-acid columns (ablation);
    ``keep_spatial=False`` removes the 3D contact edges (ablation);
    ``model_kwargs`` are forwarded to the model constructor (e.g. an
    edge-attribute-aware GAT).
    """
    set_seed(seed)
    train_d = to_pyg_data(train_g, fit_scaler(train_g, rsa_augment),
                          rsa_augment, use_edge_attr, keep_spatial)
    val_d = to_pyg_data(val_g, fit_scaler(train_g, rsa_augment),
                        rsa_augment, use_edge_attr, keep_spatial)

    in_dim = train_d[0].x.shape[1]
    if no_aa:
        # drop one-hot AA columns (0..20) from every graph at the Data level
        for d in train_d + val_d:
            d.x = d.x[:, 21:]
        in_dim = in_dim - 21

    ytr = torch.cat([d.y for d in train_d]).numpy()
    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_model(model_kind, in_dim, hidden, **(model_kwargs or {}))
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    loader = DataLoader(train_d, batch_size=batch, shuffle=True)
    history = {"train_loss": [], "val_pr_auc": [], "val_roc_auc": []}
    best_val_pr, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for data in loader:
            opt.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = loss_fn(out, data.y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        yv, pv = predict_proba(model, val_d)
        mv = evaluate(yv, pv)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_pr_auc"].append(mv["pr_auc"])
        history["val_roc_auc"].append(mv["roc_auc"])
        if mv["pr_auc"] > best_val_pr:
            best_val_pr = mv["pr_auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    yv, pv = predict_proba(model, val_d)
    best_val = evaluate(yv, pv)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "n_train_graphs": len(train_g),
        "n_val_graphs": len(val_g),
        "pos_weight": float(pos_weight),
    }


def run_split(train_g, val_g, test_g, model_kind: str, seed: int = 0,
              **kw) -> Dict:
    """Full train -> threshold-from-val -> test evaluation for one split."""
    res = train_model(train_g, val_g, model_kind, seed=seed, **kw)
    test_d = to_pyg_data(test_g,
                         fit_scaler(train_g, kw.get("rsa_augment", False)),
                         kw.get("rsa_augment", False),
                         kw.get("use_edge_attr", True),
                         kw.get("keep_spatial", True))
    if kw.get("no_aa"):
        for d in test_d:
            d.x = d.x[:, 21:]
    yt, pt = predict_proba(res["model"], test_d)
    test = evaluate(yt, pt, threshold=res["best_val"]["threshold"])
    return {"kind": model_kind, "seed": seed, **res, "test": test,
            "n_test_graphs": len(test_g)}


def stratified_graph_split(graphs: Sequence[Dict], seed: int,
                           fracs=(0.7, 0.15, 0.15)) -> Tuple[List, List, List]:
    """Random GRAPH-level split (whole conformers, never nodes).

    NOTE: despite the historical name this is a plain random permutation, not
    a stratified split. That is the correct choice here: every conformer has
    the same ~2.7% APR-positive residues, so stratifying by APR fraction would
    not change the split; the leakage-relevant property is that a conformer's
    residues never straddle train/val/test, which this guarantees.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    n1 = int(len(idx) * fracs[0])
    n2 = int(len(idx) * (fracs[0] + fracs[1]))
    return ([graphs[i] for i in idx[:n1]],
            [graphs[i] for i in idx[n1:n2]],
            [graphs[i] for i in idx[n2:]])


# ---------------------------------------------------------------------------
# Cross-ensemble transfer
# ---------------------------------------------------------------------------


def cross_ensemble_matrix(ensembles: Sequence[str], model_kind: str = "GCN",
                          seed: int = 0, out_dir: str = "outputs", **kw) -> Dict:
    """Train on each ensemble, evaluate on all three; returns PR-AUC matrix.

    NOTE: the target graphs are converted with the SAME scaler fit on the
    TRAINING ensemble - feeding raw (unscaled) continuous features into a
    model trained on z-scored features silently corrupts the transfer
    numbers (this was an early bug, see docs/PHASES_2_5_REPORT.md Phase 4).
    """
    cache = {e: load_ensemble_graphs(e, out_dir) for e in ensembles}
    rows = []
    rsa_augment = kw.get("rsa_augment", False)
    use_edge_attr = kw.get("use_edge_attr", True)
    keep_spatial = kw.get("keep_spatial", True)
    no_aa = kw.get("no_aa", False)
    for src in ensembles:
        tr, va, te = stratified_graph_split(cache[src], seed)
        r = run_split(tr, va, te, model_kind, seed=seed, **kw)
        # fit the scaler on the TRAIN split (consistent with the model that
        # run_split trained), NOT the full ensemble - a full-ensemble scaler
        # would shift the transfer features relative to the training statistics
        scaler = fit_scaler(tr, rsa_augment)
        for tgt in ensembles:
            # propagate the SAME feature/edge settings used for training; the
            # earlier version converted targets with defaults, which for
            # rsa_augment=True / no_aa=True / keep_spatial=False / edge-attr-off
            # modes silently corrupted (or crashed) the transfer evaluation
            tgt_d = to_pyg_data(cache[tgt], scaler, rsa_augment,
                                use_edge_attr, keep_spatial)
            if no_aa:
                for d in tgt_d:
                    d.x = d.x[:, 21:]
            yt, pt = predict_proba(r["model"], tgt_d)
            m = evaluate(yt, pt, threshold=r["best_val"]["threshold"])
            rows.append({"train_on": src, "test_on": tgt,
                         "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                         "f1": m["f1"]})
    return {"matrix_rows": rows, "model_kind": model_kind, "seed": seed}


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def explain_apr_nodes(model: torch.nn.Module, data: Data,
                      n_explain: int = 6, epochs: int = 200) -> Dict:
    """GNNExplainer edge masks for APR nodes of one conformer.

    Returns per-node masks with the top edges ranked, plus a summary of how
    much of the explanation mass is carried by APR-APR / APR-other edges.
    """
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type="model",
        model_config=ModelConfig(mode="binary_classification",
                                 task_level="node", return_type="raw"),
        node_mask_type="attributes",
        edge_mask_type="object",
    )
    apr_idx = torch.where(data.y > 0)[0].tolist()[:n_explain]
    out = {}
    edge_index = data.edge_index.numpy()
    for ni in apr_idx:
        expl = explainer(data.x, data.edge_index, edge_attr=data.edge_attr,
                         target=ni)
        em = expl.edge_mask.numpy()
        top = np.argsort(-em)[:10]
        # categorize the explanation edges by the APR membership of endpoints
        y = data.y.numpy()
        r1 = y[edge_index[0, top]]
        r2 = y[edge_index[1, top]]
        apr_apr = float(((r1 + r2) == 2).mean())
        apr_other = float(((r1 + r2) == 1).mean())
        frac_mass = float(em[top].sum() / max(em.sum(), 1e-12))
        out[str(ni)] = {
            "tau_resseq": int(data.tau_resseq[ni]),
            "top_edge_idx": top.tolist(),
            "top_edge_mask": em[top].tolist(),
            "top_edge_endpoints_tau": [
                [int(data.tau_resseq[edge_index[0, t]]),
                 int(data.tau_resseq[edge_index[1, t]])] for t in top],
            "frac_apr_apr": apr_apr,
            "frac_apr_other": apr_other,
            "frac_top_mass": frac_mass,
        }
    return out


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def extract_embeddings(model: torch.nn.Module, graphs: Sequence[Dict],
                       rsa_augment: bool = False, max_nodes: int = 60000
                       ) -> Dict[str, np.ndarray]:
    """Penultimate-layer node embeddings (subsampled to max_nodes)."""
    model.eval()
    scaler = fit_scaler(graphs, rsa_augment)
    datas = to_pyg_data(graphs, scaler, rsa_augment)
    emb_all, y_all, tau_all, src_all = [], [], [], []
    n = 0
    with torch.no_grad():
        for data in DataLoader(datas, batch_size=32):
            _, emb = model(data.x, data.edge_index, data.edge_attr,
                           return_emb=True)
            emb_all.append(emb.numpy())
            y_all.append(data.y.numpy())
            tau_all.append(data.tau_resseq.numpy())
            n += len(data.y)
            if n >= max_nodes:
                break
    return {"emb": np.concatenate(emb_all),
            "y": np.concatenate(y_all).astype(np.int8),
            "tau_resseq": np.concatenate(tau_all).astype(np.int32)}


# ---------------------------------------------------------------------------
# Permutation importance (MLP baseline; SHAP-analog for tabular node features)
# ---------------------------------------------------------------------------


def permutation_importance(model: torch.nn.Module, graphs: Sequence[Dict],
                           rng: np.random.Generator,
                           rsa_augment: bool = False,
                           n_repeats: int = 5,
                           groups: Optional[Dict[str, List[int]]] = None
                           ) -> Dict:
    """Drop in PR-AUC after shuffling a feature group (per-repeat)."""
    from sklearn.metrics import average_precision_score
    scaler = fit_scaler(graphs, rsa_augment)
    datas = to_pyg_data(graphs, scaler, rsa_augment)
    y, p = predict_proba(model, datas)
    base = float(average_precision_score(y, p))
    if groups is None:
        in_dim = datas[0].x.shape[1]
        groups = {"one-hot_AA": list(range(21)),
                  "hydropathy": [21], "seq_position": [22]}
        if rsa_augment:
            groups["rsa+sasa"] = [23, 24]
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = []
            for d in datas:
                x = d.x.clone()
                for c in cols:
                    x[:, c] = x[rng.permutation(len(x)), c]
                perm.append(Data(x=x, edge_index=d.edge_index,
                                 edge_attr=d.edge_attr, y=d.y,
                                 tau_resseq=d.tau_resseq))
            yp, pp = predict_proba(model, perm)
            drops.append(base - float(average_precision_score(yp, pp)))
        out[name] = {"drop_mean": float(np.mean(drops)),
                     "drop_std": float(np.std(drops)),
                     "base_pr_auc": base}
    return out
