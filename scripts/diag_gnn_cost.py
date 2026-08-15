"""Profile the true cost components of GCN training on PED00422."""
import sys
import time

import torch

sys.path.insert(0, ".")

from tau_mech.gnn import (
    fit_scaler,
    load_ensemble_graphs,
    predict_proba,
    run_split,
    stratified_graph_split,
    to_pyg_data,
    train_model,
)

torch.set_num_threads(2)

t0 = time.time()
g = load_ensemble_graphs("PED00422")
print(f"load_ensemble_graphs: {time.time() - t0:.1f}s ({len(g)} graphs)", flush=True)

tr, va, te = stratified_graph_split(g, 0)
print(f"sizes: tr={len(tr)} va={len(va)} te={len(te)}", flush=True)

t0 = time.time()
sc = fit_scaler(tr)
dtr = to_pyg_data(tr, sc)
dva = to_pyg_data(va, sc)
dte = to_pyg_data(te, sc)
print(f"to_pyg_data (all splits): {time.time() - t0:.1f}s", flush=True)

t0 = time.time()
r = run_split(tr, va, te, "GCN", seed=0, epochs=10, patience=20)
dt = time.time() - t0
print(f"run_split 10 epochs: {dt:.1f}s -> {dt / 10:.2f}s/epoch "
      f"(val PR-AUC={r['best_val']['pr_auc']:.3f})", flush=True)
