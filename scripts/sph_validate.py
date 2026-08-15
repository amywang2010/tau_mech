"""Phase 3 - SPH validation entry point.

Runs the analytic validations of the SPH engine:
  1. Couette flow  - steady-state measured ux(y) vs the analytic linear
     profile; reported as profile linearity (R^2), slope ratio (fitted /
     expected shear rate) and midpoint deviation. Run at two viscosities:
     * nu = 0.5, H = 8  (fast steady state; classic textbook config)
     * nu = 0.05, H = 4 (the STUDY solvent viscosity; short channel so the
       viscous time constant tau ~ H^2/(nu pi^2) is small enough for a
       feasible steady-state run)
  2. Laplace law   - time-averaged pressure jump across a static droplet vs
     1/R scaling (damped droplet; viscosity does not affect hydrostatics).

Usage:
    python scripts/sph_validate.py [--couette-steps N] [--laplace-steps N] [--dt F]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, validate_couette, validate_laplace  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--couette-steps", type=int, default=6000)
    p.add_argument("--laplace-steps", type=int, default=3000)
    p.add_argument("--dt", type=float, default=0.008)
    args = p.parse_args()

    params = SPHParams()
    ok_all = True

    print("=== Couette validation (steady-state linear profile) ===")
    # --- config 1: nu = 0.5, H = 8 (fast steady state) ---
    c1 = validate_couette(replace(params, mu_solvent=0.5),
                          n_steps=args.couette_steps, dt=args.dt,
                          domain=(0.0, 0.0, 24.0, 8.0))
    print(f"  [nu=0.5,  H=8] R2_fit={c1['r2_fit']:.4f} "
          f"R2_central={c1['r2_central']:.4f} "
          f"slope_ratio={c1['slope_ratio']:.4f} "
          f"slope_ratio_central={c1['slope_ratio_central']:.4f} "
          f"slip_frac={c1['slip_frac']:.3f} "
          f"(expected {c1['gamma_expected']:.4f}, got {c1['slope_fit']:.4f}) "
          f"y_mid_dev={c1['y_mid_dev']:.3f} t/tau={c1['t_over_tau']:.1f}")
    # --- config 2: nu = 0.05 (study viscosity), H = 4 ---
    c2 = validate_couette(replace(params, mu_solvent=0.05),
                          n_steps=args.couette_steps, dt=args.dt,
                          domain=(0.0, 0.0, 24.0, 4.0))
    print(f"  [nu=0.05, H=4] R2_fit={c2['r2_fit']:.4f} "
          f"R2_central={c2['r2_central']:.4f} "
          f"slope_ratio={c2['slope_ratio']:.4f} "
          f"slope_ratio_central={c2['slope_ratio_central']:.4f} "
          f"slip_frac={c2['slip_frac']:.3f} "
          f"(expected {c2['gamma_expected']:.4f}, got {c2['slope_fit']:.4f}) "
          f"y_mid_dev={c2['y_mid_dev']:.3f} t/tau={c2['t_over_tau']:.1f}")

    print("=== Laplace validation (time-averaged dP vs 1/R) ===")
    # The Laplace measurement is expensive (~1-2 h) and is performed once by
    # scripts/diag_surface_tension.py, which persists outputs/sph/
    # laplace_calibration.json. Reuse that authoritative record here so the
    # validation does not duplicate a multi-hour run; fall back to a fresh
    # measurement only when the record is missing.
    calib_path = os.path.join("outputs", "sph", "laplace_calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            import json as _json
            l = _json.load(f)
        print(f"  [reused {calib_path}]")
        # the persisted record (diag_surface_tension.py) and the raw
        # validate_laplace dict have different schemas; normalize both
        if "per_radius" in l:
            rows = {float(R): v for R, v in l["per_radius"].items()}
            sigma_fit = float(l.get("sigma_fit", float("nan")))
            lin = float(l.get("linearity_dP_vs_1R", float("nan")))
        else:
            rows = {float(k): v for k, v in l.items() if not k.startswith("_")}
            sigma_fit = float(l["_sigma_fit"])
            lin = float(l["_linearity"])
    else:
        l = validate_laplace(params, n_steps=args.laplace_steps, dt=args.dt)
        rows = {float(k): v for k, v in l.items() if not k.startswith("_")}
        sigma_fit = float(l["_sigma_fit"])
        lin = float(l["_linearity"])
    for R in sorted(rows):
        r = rows[R]
        print(f"  R={R}: dP={r['dP']:.4f}  (pin={r['pin']:.3f}, "
              f"pout={r['pout']:.3f}, n={r['n_samples']})")
    print(f"  sigma fit={sigma_fit:.4f}  linearity(dP vs 1/R)={lin:.4f}")

    # pass criteria use the CENTRAL-region fit (bulk shear rate; the full-
    # channel fit is contaminated by the ~2h wall momentum-transmission
    # layers). config 2 additionally needs t/tau > 2 to be at steady state.
    c_ok = (c1["r2_central"] > 0.99 and 0.9 < c1["slope_ratio_central"] < 1.1
            and c2["r2_central"] > 0.99
            and 0.9 < c2["slope_ratio_central"] < 1.1
            and c2["t_over_tau"] > 2.0)
    l_ok = (abs(lin) > 0.8 and sigma_fit > 0.01)
    ok_all = c_ok and l_ok
    print("VALIDATION:", "PASS"
          if ok_all else
          f"CHECK (couette_ok={c_ok}, laplace_ok={l_ok})")


if __name__ == "__main__":
    main()
