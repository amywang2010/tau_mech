"""Couette slope-ratio mechanism diagnostic (2026-09-02).

Config 1 of the Couette re-validation measured slope_ratio_central = 0.807
against the nominal-viscosity expectation. Hypothesis: the deficit is the
EFFECTIVE VISCOSITY ENHANCEMENT from the dissipative stabilizers (Monaghan
artificial viscosity + XSPH velocity smoothing), which add real shear stress
in a linear profile. If true, disabling them (pressure + Morris viscosity
only) must recover slope_ratio_central ~ 1. If the deficit persists, the
cause is elsewhere (wall coupling / kernel truncation) and must be
investigated further - a slope band would then NOT be relaxed.

Run:  python scripts/diag_couette_dissipation.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, validate_couette  # noqa: E402

OUT = Path("outputs/sph/audits/couette_dissipation_diagnostic.json")


def main():
    base = replace(SPHParams(), mu_solvent=0.5)
    configs = {
        "as_delivered (alpha=0.1, xsph=0.1)": base,
        "no_artificial_viscosity": replace(base, alpha_art=0.0),
        "no_xsph": replace(base, xsph=0.0),
        "no_dissipators (Morris only)": replace(base, alpha_art=0.0, xsph=0.0),
    }
    rows = {}
    for name, p in configs.items():
        r = validate_couette(p, n_steps=6000, dt=0.008,
                             domain=(0.0, 0.0, 24.0, 8.0))
        rows[name] = {k: r[k] for k in (
            "r2_fit", "r2_central", "slope_fit", "slope_central",
            "slope_ratio", "slope_ratio_central", "gamma_expected",
            "y_mid_dev", "u_wall_fluid", "slip_frac", "nu_used", "t_over_tau")}
        print(f"{name}: slope_ratio_central={r['slope_ratio_central']:.4f} "
              f"R2_central={r['r2_central']:.4f} y_mid_dev={r['y_mid_dev']:.4f}")
    record = {
        "hypothesis": ("slope deficit = effective-viscosity enhancement "
                        "from Monaghan AV + XSPH; disabling them must "
                        "recover slope_ratio_central ~ 1"),
        "decision_rule": ("if the no-dissipator config recovers ~1.0, the "
                           "deficit is ATTRIBUTED and the acceptance "
                           "framework reports nu_eff as a measured property "
                           "(the sweep already uses the measured local shear "
                           "rate); if not, the band is NOT relaxed and the "
                           "cause is investigated further"),
        "configs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
