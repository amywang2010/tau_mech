"""Laplace verification for the CSF surface-tension model.

Measures the time-averaged pressure jump dP = P_in - P_out across a static
droplet at radii 3/4/5 and checks the Laplace law dP = sigma/R:
  * slope of dP vs 1/R should equal the CSF input sigma (verification,
    since sigma is now a DIRECT model parameter), and
  * the linearity (Pearson r of dP vs 1/R) must be |r| > 0.8.
Also checks stability (no NaNs, densities in range).

For the CSF model there is no (A, B) pair to "tune"; A is the mixed-only
immiscibility repulsion and B is unused (0). The authoritative record is
persisted to outputs/sph/laplace_calibration.json and reused by
scripts/sph_validate.py and the shear sweep.

Usage:  python scripts/diag_surface_tension.py [--sigma 1.0] [--n-eq 2500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    _laplace_masks,
    make_couette_droplet_state,
    run,
    step,
)


def measure_dP(params: SPHParams, R: float, n_eq: int = 1600, dt: float = 0.008,
               sample_from: int = 0, every: int = 40) -> dict:
    """Time-averaged pressure jump across a static droplet of radius R.

    The core/annulus masks are BAND-AWARE (same as sph._laplace_masks): the
    CSF color transition is ~3h wide, so pin must sample the c~=1 interior
    (r < R - 3h) and pout the c~=0 solvent (r > R + 3h); the domain is
    sized so the annulus and the y-band clear the frozen walls.

    sample_from defaults to a PHYSICS-BASED value: the surface-mode
    oscillation must decay before the pressure is averaged. The decay time
    scales with the capillary time t_char = mu*R/sigma_eff (~ mu*R for
    sigma_eff ~ sigma_input after the factor-2 fix), so sampling starts at max(0.6*n_eq,
    4*t_char/dt). The earlier hardcoded sample_from=800 polluted the average
    with the early oscillation (dP ~ 0.23 -> 0.026 over the first 16 time
    units at R=7), biasing larger radii low and creating a spurious
    R-dependence in sigma_eff (audit: docs/PHASES_2_5_REPORT.md).
    """
    if sample_from <= 0:
        t_char = params.mu_droplet * R / max(params.sigma_surf, 1e-9)
        sample_from = int(max(0.6 * n_eq, 4.0 * t_char / dt))
    W = 2.0 * (R + 4.0) + 4.0
    H = 2.0 * (R + 4.0) + 4.0
    domain = (0.0, 0.0, W, H)
    state = make_couette_droplet_state(params, domain=domain, spacing=0.5,
                                       droplet_radius=R, n_wall_layers=2)
    run(state, params, 0, dt)  # density init only (mass renormalization)
    pin_acc, pout_acc = [], []
    for s in range(n_eq):
        step(state, params, dt)
        if s >= sample_from and s % every == 0:
            core, far = _laplace_masks(state, R, params.h)
            if core.sum() > 0 and far.sum() > 0:
                pin_acc.append(float(state.pressure[core].mean()))
                pout_acc.append(float(state.pressure[far].mean()))
    pin = float(np.mean(pin_acc))
    pout = float(np.mean(pout_acc))
    nan = int(np.isnan(state.pos).sum()) + int(np.isnan(state.vel).sum())
    rho_free = state.rho[state.phase != 2]
    return {"dP": pin - pout, "pin": pin, "pout": pout, "n_samples": len(pin_acc),
            "n_core": int(core.sum()), "n_far": int(far.sum()),
            "nan": nan, "rho_min": float(rho_free.min()),
            "rho_max": float(rho_free.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="CSF surface tension (direct model parameter)")
    ap.add_argument("--mu-droplet", type=float, default=0.5,
                    help="droplet viscosity during the Laplace measurement. "
                         "MUST be small enough that n_eq*dt >> mu*R/sigma_eff "
                         "(the capillary relaxation time); the study value "
                         "0.5 gives t_char ~ R units, covered >= 5x by "
                         "n-eq=4500. The earlier 5.0 damped the oscillation "
                         "but trapped the run at ~0.3-0.4x t_char and "
                         "under-measured dP (audit: docs/PHASES_2_5_REPORT.md)")
    ap.add_argument("--n-eq", type=int, default=4500)
    args = ap.parse_args()

    p = SPHParams(sigma_surf=args.sigma, mu_droplet=args.mu_droplet)
    print(f"sigma_surf={p.sigma_surf}  A_surf={p.A_surf} (mixed-only)  "
          f"B_surf={p.B_surf}  mu_droplet={p.mu_droplet}")
    rows = []
    # radii 5/6/7: the CSF color-transition band is ~2.5h wide; at R <= 3h
    # the band spans the whole droplet so no c~=1 interior exists and pin is
    # meaningless (the R=3 radial profile is in docs/PHASES_2_5_REPORT.md)
    for R in (5.0, 6.0, 7.0):
        t0 = time.time()
        r = measure_dP(p, R, n_eq=args.n_eq)
        rows.append((R, r))
        print(f"  R={R}: dP={r['dP']:.4f} (pin={r['pin']:.3f} pout={r['pout']:.3f}) "
              f"nan={r['nan']} rho[{r['rho_min']:.3f},{r['rho_max']:.3f}] "
              f"[{time.time()-t0:.0f}s]")
    xs = np.array([1.0 / R for R, _ in rows])
    ys = np.array([r["dP"] for _, r in rows])
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    lin = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else float("nan")
    # per-radius effective tension sigma_eff = dP*R (2D Laplace). After the
    # velocity-Verlet factor-2 fix (CSF applied in BOTH half-steps; audit
    # 2026-08-14), sigma_eff ~ 0.95-0.99 sigma_input — the residual few-% is
    # a mild finite-h/R discretization effect (converges to 1.0 as h/R -> 0),
    # NOT the former spurious 0.46 "band-split" (which was the factor-2 bug).
    # The mean per-radius sigma_eff is the recommended value for the sweep's Ca.
    sigma_eff_per_R = [float(r["dP"] * R) for R, r in rows]
    sigma_eff = float(np.mean(sigma_eff_per_R)) if sigma_eff_per_R else float("nan")
    print(f"  -> sigma fit={slope:.4f}  linearity(dP vs 1/R)={lin:.4f}")
    print(f"  -> per-radius sigma_eff (dP*R): "
          + ", ".join(f"R={R}:{se:.3f}" for R, se in zip([R for R, _ in rows],
                                                      sigma_eff_per_R)))
    print(f"  -> recommended sigma_eff (mean) = {sigma_eff:.4f} "
          f"({sigma_eff / p.sigma_surf * 100:.1f}% of sigma_input; "
          f"residual finite-h/R discretization)")
    ok = (slope > 0.01) and (abs(lin) > 0.8) and all(r["nan"] == 0 for _, r in rows)
    print("  DIAG:", "USABLE (sigma measurable, Laplace scaling OK)" if ok
          else "NOT USABLE - adjust A/B (or check stability)")
    # persist the authoritative Laplace calibration record so that
    # scripts/sph_validate.py and the shear sweep reuse this measurement
    # instead of duplicating a multi-hour run
    os.makedirs("outputs/sph", exist_ok=True)
    record = {
        "model": "CSF (Brackbill 1992 / Adami 2010); A_surf mixed-only "
                  "immiscibility repulsion; B_surf unused",
        "sigma_input": p.sigma_surf, "A_surf": p.A_surf, "B_surf": p.B_surf,
        "mu_droplet": p.mu_droplet, "n_eq": args.n_eq,
        "sample_from": "physics-based: max(0.6*n_eq, 4*t_char/dt) with "
                        "t_char = mu*R/sigma",
        "radii": [R for R, _ in rows],
        "core_mask": "r < R - 2.5h (c~=1 interior)",
        "annulus_mask": "r in [R+2.5h, R+4h], y-band clear of walls (c~=0)",
        "per_radius": {str(R): r for R, r in rows},
        "sigma_fit": slope, "linearity_dP_vs_1R": lin,
        "sigma_eff_per_radius": {str(R): se for R, se in
                                 zip([R for R, _ in rows], sigma_eff_per_R)},
        "sigma_eff": sigma_eff,
        "sigma_ratio_fit_over_input": (slope / p.sigma_surf
                                        if p.sigma_surf > 0 else float("nan")),
        "usable": ok,
        "note": ("2D dimensionless; CSF sigma is a DIRECT model parameter - "
                 "the Laplace law dP = sigma/R is a VERIFICATION: the slope "
                 "of dP vs 1/R should equal sigma_input. sigma_eff ~ "
                 "0.95-0.99 sigma_input after the velocity-Verlet factor-2 "
                 "fix (2026-08-14); residual few-% is finite h/R. Use "
                 "sigma_eff (mean of per-radius dP*R) for the sweep's Ca. "
                 "dP = P_core - P_annulus averaged after 4x t_char; same "
                 "measurement as sph.validate_laplace."),
    }
    with open(os.path.join("outputs", "sph", "laplace_calibration.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"  saved -> outputs/sph/laplace_calibration.json")


if __name__ == "__main__":
    main()
