"""Phase 1 preprocessing entry point.

Usage (from the tau_mech/ directory, with the venv active):

    python scripts/preprocess.py --all                  # all ensembles, full
    python scripts/preprocess.py --ensemble PED00192    # one ensemble, full
    python scripts/preprocess.py --ensemble PED00422 --n-models 20  # subset

Outputs land in outputs/<PED_ID>/ (see pipeline.py for the schema).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.config import PipelineConfig          # noqa: E402
from tau_mech.constants import ENSEMBLES, REFERENCE_SASA  # noqa: E402
from tau_mech.features import GraphConfig           # noqa: E402
from tau_mech.pipeline import process_ensemble     # noqa: E402
from tau_mech.provenance import write_provenance   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess PED Tau ensembles.")
    p.add_argument("--ensemble", action="append", choices=list(ENSEMBLES.keys()),
                   help="PED ID(s) to process (repeatable). Default: all.")
    p.add_argument("--all", action="store_true", help="process all ensembles")
    p.add_argument("--n-models", type=int, default=None,
                   help="process only this many conformers per ensemble")
    p.add_argument("--start-at", type=int, default=0,
                   help="skip this many conformers at the start")
    p.add_argument("--data-dir", default="..", help="directory with the raw *.tar.gz")
    p.add_argument("--out-dir", default="outputs", help="output directory")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="heavy-atom edge cutoff (Angstrom)")
    p.add_argument("--no-sequential", action="store_true",
                   help="do not add sequential backbone edges")
    p.add_argument("--seq-adjacency", type=int, default=2)
    p.add_argument("--probe", type=float, default=1.4, help="SASA probe radius (A)")
    p.add_argument("--n-points", type=int, default=480, help="SASA probe points/atom")
    p.add_argument("--heavy", action="store_true", help="keep hydrogens (default: drop)")
    p.add_argument("--no-save-models", action="store_true",
                   help="skip per-conformer graph .npz files")
    p.add_argument("--no-contact-maps", action="store_true",
                   help="skip consolidated contact maps (reduces output size)")
    args = p.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        heavy_only=not args.heavy,
        probe_radius=args.probe,
        n_probe_points=args.n_points,
        n_models=args.n_models,
        start_at=args.start_at,
        ensemble_ids=args.ensemble,
        graph=GraphConfig(
            edge_cutoff=args.cutoff,
            add_sequential=not args.no_sequential,
            seq_adjacency=args.seq_adjacency,
        ),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg_doc = cfg.to_dict()
    # record the exact reference (max) SASA table used for rASA, so every
    # numeric claim is reproducible from the config record alone
    cfg_doc["rsa_reference"] = {
        "source": ("Tien M.Z. et al. 2013, PLoS ONE 8(11):e80635; Table 1 "
                   "THEORETICAL scale, ALLOWED Ramachandran region "
                   "(Gly-X-Gly tripeptides, Lee-Richards/DSSP, probe 1.4 A)"),
        "max_asa_angstrom2": dict(REFERENCE_SASA),
    }
    with open(os.path.join(cfg.output_dir, "config_used.json"), "w") as f:
        json.dump(cfg_doc, f, indent=2)

    write_provenance(cfg.data_dir, cfg.output_dir)

    targets = cfg.resolved_ensembles()
    if args.all and args.ensemble is None:
        targets = dict(ENSEMBLES)
    elif args.ensemble:
        targets = {k: ENSEMBLES[k] for k in args.ensemble}
    else:
        targets = dict(ENSEMBLES)

    summaries = []
    for eid in targets:
        print(f"=== processing {eid} ===")
        s = process_ensemble(
            cfg,
            eid,
            save_models=not args.no_save_models,
            save_contact_maps=not args.no_contact_maps,
        )
        summaries.append(s)

    with open(os.path.join(cfg.output_dir, "processing_report.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print("Done. Reports in", os.path.abspath(cfg.output_dir))


if __name__ == "__main__":
    main()
