"""Print a compact summary of the processed ensembles and compare key
descriptors with published experimental values (SAXS-derived Rg).

Usage:
    python scripts/summarize.py [--out-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Published experimental anchors (with citations) used to sanity-check the
# computed ensembles. Values depend on buffer conditions; a +/-10% tolerance
# band is used because the anchors are themselves approximate across studies.
EXPERIMENTAL_RG = {
    "PED00422": {"target": 67.0, "tolerance_pct": 10.0,
                 "ref": "Tau-441 SAXS Rg 6.5-6.7 nm (He et al. 2022, ACS Chem. "
                        "Neurosci.; SASBDB SASDLU4: 6.9 nm)"},
    "PED00192": {"target": 38.0, "tolerance_pct": 10.0,
                 "ref": "K18 SAXS Rg ~3.8 nm (Mukrasch et al. 2005, J. Biol. "
                        "Chem.; He et al. 2024, J. Chem. Inf. Model.)"},
    "PED00443": {"target": 38.0, "tolerance_pct": 10.0,
                 "ref": "K18 SAXS Rg ~3.8 nm (see PED00192)"},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    print("=" * 78)
    print("Tau mechanobiology - Phase 1 processing summary")
    print("=" * 78)
    for eid in ["PED00422", "PED00192", "PED00443"]:
        spath = os.path.join(args.out_dir, eid, "summary.json")
        if not os.path.exists(spath):
            print(f"\n{eid}: not processed yet (missing {spath})")
            continue
        with open(spath) as f:
            s = json.load(f)
        st = s["statistics"]
        print(f"\n{eid}  ({s['construct']})")
        print(f"  method               : {s['method']}")
        print(f"  conformers           : {s['n_models_processed']} "
              f"(expected {s['n_models_expected']})  residues: {s['n_residues']}")
        print(f"  APR motifs (Tau num) : {s['motif_spans_tau_numbering']}")
        r = lambda k: (st[k]["mean"], st[k]["std"])  # noqa: E731
        m1, s1 = r("rg_mass_weighted")
        m2, s2 = r("rg_equal_weight")
        m3, s3 = r("end_to_end")
        print(f"  Rg mass-weighted     : {m1:6.1f} +/- {s1:5.1f} A")
        print(f"  Rg equal-weight      : {m2:6.1f} +/- {s2:5.1f} A")
        print(f"  end-to-end           : {m3:6.1f} +/- {s3:5.1f} A")
        exp = EXPERIMENTAL_RG.get(eid)
        if exp:
            tgt = exp["target"]
            tol = exp["tolerance_pct"]
            lo, hi = tgt * (1 - tol / 100), tgt * (1 + tol / 100)
            ratio = m1 / tgt
            status = "OK" if lo <= m1 <= hi else f"OUTSIDE {tol:.0f}% band (ratio {ratio:.2f})"
            print(f"  experimental anchor  : ~{tgt:.0f} A  (computed {m1:.1f}, ratio {ratio:.2f}) "
                  f"{status}")
            print(f"                        {exp['ref']}")
        m4, s4 = r("apr1_vqiink_mean_rsa")
        m5, s5 = r("apr2_vqivyk_mean_rsa")
        print(f"  APR1 VQIINK rSA      : {m4:.3f} +/- {s4:.3f}")
        print(f"  APR2 VQIVYK rSA      : {m5:.3f} +/- {s5:.3f}")
        m6, s6 = r("mean_degree")
        print(f"  graph mean degree    : {m6:5.1f} +/- {s6:4.1f}  "
              f"(cutoff {s['graph_config']['edge_cutoff']} A)")
    print("\nSee outputs/<PED_ID>/ for per-conformer data.")


if __name__ == "__main__":
    main()
