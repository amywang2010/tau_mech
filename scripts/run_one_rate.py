"""Run ONE sweep rate in an isolated out_dir (parallel-wave runner).

The sweep module's resume/persistence is single-writer per out_dir; running
rates as separate processes in separate out_dirs (calibration copied in)
allows safe parallelism. Records are merged into the canonical sweep file
afterwards by scripts/merge_final_sweep.py.

Usage: python scripts/run_one_rate.py <rate> <out_dir> [--eq N] [--shear N]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, droplet_shear_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate", type=float)
    ap.add_argument("out_dir", type=str)
    ap.add_argument("--eq", type=int, default=4000)
    ap.add_argument("--shear", type=int, default=50765)
    args = ap.parse_args()
    params = SPHParams(mu_solvent=1.0, mu_droplet=10.0)
    rows = droplet_shear_sweep(
        params, shear_rates=[args.rate], eq_steps=args.eq,
        shear_steps=args.shear, dt=0.008, spacing=0.5,
        droplet_radius=3.0, domain=(0.0, 0.0, 24.0, 16.0),
        out_dir=args.out_dir,
    )
    print(f"DONE rate={args.rate} rows={len(rows)}")


if __name__ == "__main__":
    main()
