"""Aggregate GNNExplainer top-edge endpoints by Tau region (report helper)."""
import json
from collections import Counter

x = json.load(open("outputs/gnn/explainability.json"))
regions = Counter()
n_nodes = 0
for g in x["graphs"]:
    for node, nd in g["nodes"].items():
        n_nodes += 1
        for ep in nd["top_edge_endpoints_tau"]:
            a = ep[0]  # first endpoint residue (Tau numbering)
            if 244 <= a <= 274:
                regions["R1 (244-274)"] += 1
            elif 275 <= a <= 305:
                regions["R2 (275-305)"] += 1
            elif 306 <= a <= 336:
                regions["R3 (306-336)"] += 1
            elif 337 <= a <= 368:
                regions["R4 (337-368)"] += 1
            elif 369 <= a <= 441:
                regions["C-term (369-441)"] += 1
            else:
                regions["N-term (<244)"] += 1
tot = sum(regions.values())
print(f"explained nodes: {n_nodes}  top edges: {tot}")
for k, v in regions.most_common():
    print(f"  {k:18s} {v:4d}  ({100 * v / tot:.1f}%)")
