"""Phase 2 - Exploratory Data Analysis entry point.

Usage (from tau_mech/):
    python scripts/eda.py [--out-dir outputs] [--fig-dir outputs/figures]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.eda import run_eda, print_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 EDA for Tau ensembles.")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--ensemble", action="append", default=None,
                   choices=["PED00422", "PED00192", "PED00443"])
    args = p.parse_args()
    report = run_eda(out_dir=args.out_dir, fig_dir=args.fig_dir,
                     ensemble_ids=args.ensemble)
    print_report(report)


if __name__ == "__main__":
    main()
