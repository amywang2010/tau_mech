"""Final SPH shear sweep with the fully validated solver (2026-09-02).

Pre-registered design (fixed BEFORE any result was seen):

* Solver state: CSF symmetric-stencil fix, verified by the four-probe CSF
  symmetry audit, permutation invariance to machine precision, conservative
  internal force (net force 0.05%), and the full-duration zero-shear gate
  (all six pre-registered criteria PASS; outputs/sph/audits/
  zero_shear_baseline.json).
* Surface tension: fresh Laplace calibration on the fixed solver
  (sigma_eff = 1.0641, linearity 0.9999, R = 5/6/7; the per-radius values
  decrease toward sigma_input as h/R -> 0, documented discretization).
* Viscosities: mu_solvent = 1.0, mu_droplet = 10.0 (lambda = 10 preserved,
  matching the recorded physiological mapping rationale).
* Rates: 0.0 (no-shear CONTROL - identical protocol, zero wall speed),
  0.001, 0.003, 0.01, 0.03, 0.1 (dimensionless), covering the physiological
  Ca window via the analytic Taylor limit and the numerically accessible
  window with measured local shear rates.
* Run length per rate: eq_steps = 4000 (the gate protocol) + shear phase
  50765 steps (~406 time units), identical to the zero-shear gate window so
  the control and sheared cases are directly comparable and the D(t) plateau
  fit has >= 5 capillary times (t_char ~ 3).
* Acceptance (pre-registered): every sheared case must show
  (a) monotone-in-time deformation toward the fitted plateau with fit R2 >=
  0.8, (b) Ca_measured consistent with Ca_nominal within the documented wall
  slip factor, (c) D_inf increasing with Ca, and the zero-shear control must
  satisfy the same bounded-deformation gate as the baseline audit (max
  |D - D0| < 0.02 after equilibration).
* The defective-solver Aug-15 sweep is archived under
  outputs/sph/archive_pre_csffix/ and is NOT comparable (one-sided curvature
  stencil); the resume logic cannot reach it.

Usage:
    python scripts/run_final_sweep.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, droplet_shear_sweep  # noqa: E402

OUT = Path("outputs/sph/sph_shear_sweep.json")


def main() -> None:
    smoke = "--smoke" in sys.argv
    params = SPHParams(mu_solvent=1.0, mu_droplet=10.0)
    rates = [0.001, 0.003] if smoke else [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    eq_steps, shear_steps = (500, 1500) if smoke else (4000, 50765)
    rows = droplet_shear_sweep(
        params, shear_rates=rates, eq_steps=eq_steps,
        shear_steps=shear_steps, dt=0.008, spacing=0.5,
        droplet_radius=3.0, domain=(0.0, 0.0, 24.0, 16.0),
        out_dir="outputs/sph",
    )
    if smoke:
        print("SMOKE OK (sweep harness runs end-to-end)")
        return
    # Serial mode persists the canonical sweep file; the acceptance summary
    # is owned exclusively by scripts/merge_final_sweep.py (single writer).
    print("serial sweep complete; run scripts/merge_final_sweep.py for the "
          "pre-registered acceptance checks")


if __name__ == "__main__":
    main()
