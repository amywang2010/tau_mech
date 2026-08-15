"""Smoke test for the GNN module (small subset, few epochs)."""
import sys
import time

sys.path.insert(0, ".")
from tau_mech.gnn import (  # noqa: E402
    _PYG_IMPORT_ERROR, explain_apr_nodes, extract_embeddings,
    load_ensemble_graphs, run_split, stratified_graph_split, to_pyg_data,
)

print("PyG import error:", _PYG_IMPORT_ERROR)
g = load_ensemble_graphs("PED00192")[:30]
print("loaded", len(g), "graphs; node feat dim", g[0]["node_features"].shape,
      "| edges", g[0]["edge_index"].shape)
tr, va, te = stratified_graph_split(g, 0)
print("split sizes", len(tr), len(va), len(te))
t0 = time.time()
r = run_split(tr, va, te, "GCN", seed=0, epochs=4, patience=10)
print(f"train ok in {time.time()-t0:.1f}s; "
      f"val PR-AUC={r['best_val']['pr_auc']:.3f} "
      f"test PR-AUC={r['test']['pr_auc']:.3f}")
t0 = time.time()
emb = extract_embeddings(r["model"], te[:5], max_nodes=5000)
print(f"embeddings ok {emb['emb'].shape} in {time.time()-t0:.1f}s")
t0 = time.time()
d = to_pyg_data([te[0]])[0]
ex = explain_apr_nodes(r["model"], d, n_explain=2, epochs=20)
print(f"explain ok in {time.time()-t0:.1f}s:", list(ex.keys()))
for k, v in ex.items():
    print("  node", k, "tau", v["tau_resseq"],
          "frac_apr_apr", round(v["frac_apr_apr"], 2),
          "frac_top_mass", round(v["frac_top_mass"], 2))
print("SMOKE OK")
