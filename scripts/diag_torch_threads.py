"""Time a few GCN epochs at different torch thread counts.

For tiny graphs (441 nodes) thread-launch overhead can dominate, so the
fastest config is not necessarily the one with the most threads.
"""
import sys
import time

import torch

sys.path.insert(0, ".")

from tau_mech.gnn import load_ensemble_graphs, run_split, stratified_graph_split

g = load_ensemble_graphs("PED00192")
for nt in (1, 2, 4, 8):
    torch.set_num_threads(nt)
    tr, va, te = stratified_graph_split(g, 0)
    t0 = time.time()
    r = run_split(tr, va, te, "GCN", seed=0, epochs=3, patience=20)
    pv = r["best_val"]["pr_auc"]
    print(f"threads={nt}: 3 epochs in {time.time() - t0:.1f}s "
          f"(val PR-AUC={pv:.3f})", flush=True)
