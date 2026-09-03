"""Couette wall-slip resolution study, v2 (2026-09-03).

WHY v2 (honest protocol history)
--------------------------------
v1 (2026-09-02, archived as couette_resolution_study_v1_fixedh.json)
varied the lattice spacing while keeping the kernel radius at its default
h = 1.0 for all three levels. Its own pre-registered decision rule, however,
defines refinement as "h = 2*spacing -> 0 at fixed physical geometry" - i.e.
kernel and spacing must be refined TOGETHER. With h frozen, the ~2h-thick
wall momentum-transmission layer was identical in ABSOLUTE units across all
three v1 runs; the three runs sampled the same physical boundary layer at
different particle densities. v1 therefore could not decide attribution
branch (a) vs (b): its slip values (0.852, 0.754, 0.814) are sampling noise
around ~0.80, not a convergence trend, and its non-monotone scatter is
exactly what that design predicts. v1 is RETAINED as a fixed-h control (it
demonstrates the measured slip is not an artifact of particle density at
fixed kernel) and its coarse level doubles as v2's regression anchor.

v2 design (the study the pre-registration specifies)
----------------------------------------------------
* Co-refinement: h = 2*spacing at every level (h/dx = 2, the standard SPH
  convergence design). All h-denominated model parameters (wall layers,
  central zone, repulsion/attraction cutoffs, switch width) are defined in
  units of h in the solver, so the physical configuration is IDENTICAL
  across levels; only the discretization varies. dt scales WITH h
  (dt = 0.008 * h), pinning the Courant number at 0.08 - the value at
  which EVERY prior validation of this solver ran - so Reynolds (U*H/nu
  = 346) and Courant numbers are identical at all levels and the ONLY
  variable across levels is h. Window step counts scale accordingly:
  5700 / 8550 / 11400 steps for the same simulated time t = 45.6.
* Geometry: same exact-row domains as v1 (fluid rows 19/29/39 at spacings
  0.5 / 1/3 / 0.25; no-slip planes one row outside the fluid).
* Quasi-steady guard (replaces v1's ambiguous viscous-time bookkeeping):
  each level runs a 5700-step window (t = 45.6, the length at which all
  three v1 levels MEASURED quasi-steady profiles, r2_central >= 0.9993),
  then an independent 8550-step (1.5x) run. Steadiness criterion:
  |slip(8550) - slip(5700)| <= 0.005 and r2_central >= 0.99 on both. On
  failure the window escalates once (12825 steps); if steadiness still
  fails, the level - and the study - ABORT with an explicit record. An
  unsteady slip value must never enter the pre-registered attribution.
  Momentum-diffusion times at the nominal nu = 0.05 are REPORTED per
  level for transparency, not used to pick the window.
* Regression anchor: the level-1 first window (spacing 0.5, h = 1.0,
  5700 steps) is bit-identical to v1's coarse run (same solver state, same
  IC, deterministic integrator). slip_frac must reproduce v1's 0.8520350
  to ~1e-9; any mismatch is solver nondeterminism and fails the study.
* Decision rule (unchanged from v1, now decisive):
  (a) slip_frac decreases monotonically with refinement (fit over the
      three levels) => boundary-discretization artifact: documented, the
      sweep protocol's use of the MEASURED local shear rate stands;
  (b) slip_frac resolution-independent => wall-coupling FORMULATION
      property; the acceptance band is NOT relaxed either way.

Cost: ~6 h uncontended serial (particle-steps: 5700x912 + 8550x2088 +
11400x3744 at the measured 3.3e-4 s/particle-step); one escalation on
every level roughly doubles the finer levels (~12 h worst case).

Run:  python scripts/diag_couette_resolution.py [--smoke]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, validate_couette  # noqa: E402

OUT = Path("outputs/sph/audits/couette_resolution_study.json")
V1_BACKUP = Path("outputs/sph/audits/couette_resolution_study_v1_fixedh.json")
SPACING_ROWS = {0.5: 19, 1.0 / 3.0: 29, 0.25: 39}  # exact fluid rows per level
BASE_TIME = 45.6           # base window length (all v1 levels steady here);
                           # per-level step count = BASE_TIME / dt(h)
STEADY_TOL = 0.005         # max |slip(1.5T) - slip(T)| accepted as steady
R2_MIN = 0.99              # linearity floor on every window
MAX_ESCALATIONS = 1        # window escalation cap (1.5x, once, then abort)
V1_COARSE_SLIP = 0.8520350443108875  # regression anchor (v1 record, spacing 0.5)


def domain_for(spacing: float, n_rows: int):
    """Exact-row Couette domain: fluid occupies exactly n_rows lattice rows.

    Rows sit at y = k*dy (the lattice generator anchors the first fluid row at
    y0 = 0 up to the documented ~1e-16 accumulation drift). Choosing
    y1 = (n_rows - 0.5)*dy places the top boundary strictly between rows
    n_rows-1 and n_rows, so exactly n_rows rows are fluid for any spacing.
    H_wall = (n_rows + 1)*dy (bottom/top no-slip planes one row outside).
    """
    dy = spacing * np.sqrt(3.0) / 2.0
    y1 = (n_rows - 0.5) * dy
    return (0.0, 0.0, 24.0, float(y1))


def run_level(spacing: float, n_rows: int, smoke: bool):
    """One resolution level: h = 2*spacing, guarded quasi-steady windows."""
    h = 2.0 * spacing
    params = SPHParams(h=h)  # every other parameter: as-delivered defaults;
    # all h-denominated parameters (r_rep, r_att, switch_delta, wall layers,
    # central zone) scale with h inside the solver, so the PHYSICAL config
    # is identical across levels - only the discretization varies.
    domain = domain_for(spacing, n_rows)
    H_wall = (n_rows + 1) * spacing * np.sqrt(3.0) / 2.0
    dt = 0.008 * h  # dt proportional to h: CFL = c_s*dt/h pinned at 0.08,
    # the value at which every prior validation of this solver ran - so
    # Re and CFL are identical at all levels and only h varies.
    nu = params.mu_solvent / 1.0  # rho0 = 1; nominal kinematic viscosity
    tau_nom = H_wall**2 / (nu * np.pi**2)  # reported, NOT used to size runs
    cfl = dt * params.c_s / h
    visc_ratio = dt * nu / (0.125 * h**2)  # viscous stability margin

    if smoke:
        n_steps = 120
        r = validate_couette(params=params, n_steps=n_steps, dt=dt,
                             domain=domain, spacing=spacing)
        r.update({"h": h, "spacing": spacing, "n_rows": n_rows,
                  "H_wall": H_wall, "n_steps": n_steps})
        return r, {"n_steps": n_steps, "steady": None, "escalations": 0}

    windows, escalations, steady = [], 0, False
    n_steps = int(np.ceil(BASE_TIME / dt))  # t = 45.6 at EVERY level
    results = {}
    while True:
        r = validate_couette(params=params, n_steps=n_steps, dt=dt,
                             domain=domain, spacing=spacing)
        results[n_steps] = r
        windows.append(n_steps)
        if len(results) >= 2:
            a, b = results[windows[-2]], results[windows[-1]]
            drift = abs(b["slip_frac"] - a["slip_frac"])
            ok = (drift <= STEADY_TOL
                  and a["r2_central"] >= R2_MIN and b["r2_central"] >= R2_MIN)
            if ok:
                steady = True
                break
            if escalations >= MAX_ESCALATIONS:
                break
            escalations += 1
            n_steps = int(np.ceil(n_steps * 1.5))
        else:
            n_steps = int(np.ceil(n_steps * 1.5))

    final = results[windows[-1]]
    guard = {
        "windows": windows,
        "steady": steady,
        "escalations": escalations,
        "first_window_slip": results[windows[0]]["slip_frac"],
        "slip_drift_last": (abs(results[windows[-1]]["slip_frac"]
                                - results[windows[-2]]["slip_frac"])
                            if len(windows) >= 2 else None),
        "r2_central_all": {str(k): results[k]["r2_central"]
                           for k in windows},
        "note": ("quasi-steady guard: two consecutive windows with "
                 "|dslip| <= 0.005 and r2_central >= 0.99; the headline row "
                 "is the FINAL window"),
    }
    final.update({"h": h, "spacing": spacing, "n_rows": n_rows,
                  "H_wall": H_wall, "tau_nu_nominal": tau_nom,
                  "t_over_tau_nu_nominal": final["t_sim"] / tau_nom,
                  "cfl_dt_ratio": cfl, "viscous_dt_ratio": visc_ratio,
                  "n_steps": final["n_steps"]})
    return final, guard


def level1_anchor(guard: dict) -> dict:
    """Regression anchor: level 1 FIRST window (spacing 0.5, h=1.0,
    dt=0.008, 5700 steps) must reproduce the v1 coarse run bit-for-bit
    (deterministic solver). Level 1 uses the same h, dt and window as v1's
    coarse run, so the FIRST window is the anchor - not the final one -
    because v1's 0.8520 was measured at exactly t = 45.6; later (longer)
    windows may legitimately drift by up to the steady tolerance. Checked
    immediately after level 1 so a determinism failure cannot waste the
    remaining compute budget.
    """
    return {
        "v1_slip": V1_COARSE_SLIP,
        "v2_slip": guard["first_window_slip"],
        "abs_diff": abs(guard["first_window_slip"] - V1_COARSE_SLIP),
        "pass": bool(abs(guard["first_window_slip"] - V1_COARSE_SLIP) < 1e-9),
        "note": ("level-1 FIRST-window (5700-step) replication of the v1 "
                 "coarse run (same state/IC/deterministic integrator); "
                 "failure would indicate solver nondeterminism and "
                 "invalidate the study"),
    }


def main() -> None:
    smoke = "--smoke" in sys.argv

    # Archive the v1 (fixed-h) record exactly once, before v2 overwrites it.
    if OUT.exists() and not V1_BACKUP.exists():
        shutil.copy2(OUT, V1_BACKUP)
        print(f"v1 (fixed-h) record archived -> {V1_BACKUP}")

    rows = {}
    guards = {}
    anchor = None
    for spacing, n_rows in sorted(SPACING_ROWS.items(), reverse=True):
        r, guard = run_level(spacing, n_rows, smoke)
        key = f"s_{spacing:.4f}_h_{2.0 * spacing:.4f}_rows_{n_rows}"
        rows[key] = r
        guards[key] = guard
        print(f"s={spacing:.4f} h={2.0 * spacing:.3f} rows={n_rows}: "
              f"slip_frac={r['slip_frac']:.4f} "
              f"u_wall_fluid={r['u_wall_fluid']:.4f} "
              f"slope_ratio_central={r['slope_ratio_central']:.4f} "
              f"R2c={r['r2_central']:.4f} steady={guard['steady']} "
              f"[{r['n_steps']} steps]")
        if smoke:
            print("SMOKE OK (no record written)")
            return
        if not guard["steady"]:
            # An unsteady measurement must never enter the attribution:
            # abort the study with an explicit, attributable record.
            print("QUASI-STEADY GUARD FAILED - study aborted; no "
                  "attribution written (unsteady slip must not inform "
                  "the decision rule).")
            OUT.write_text(json.dumps({
                "study_version": "v2 (2026-09-03)",
                "level": key,
                "steady_guards": guards,
                "configs": {key: r},
                "attribution": None,
                "aborted": f"quasi-steady guard failed at level {key}",
            }, indent=2))
            return
        if spacing == 0.5:
            # Early determinism gate: abort BEFORE levels 2-3 if the
            # anchor fails (saves ~4 h of doomed compute).
            anchor = level1_anchor(guard)
            print("REGRESSION ANCHOR:", json.dumps(anchor, indent=1))
            if not anchor["pass"]:
                print("REGRESSION ANCHOR FAILED - solver nondeterminism; "
                      "study aborted, no attribution written.")
                OUT.write_text(json.dumps({
                    "study_version": "v2 (2026-09-03)",
                    "regression_anchor": anchor,
                    "attribution": None,
                    "aborted": "regression anchor failed at level 1",
                }, indent=2))
                return

    # Regression anchor (already checked early after level 1; recorded in
    # the final record for the permanent evidence chain).
    anchor = anchor or level1_anchor(guards["s_0.5000_h_1.0000_rows_19"])

    # Pre-registered attribution analysis: slip vs h = 2*spacing.
    h = np.array([r["h"] for r in rows.values()])
    slip = np.array([r["slip_frac"] for r in rows.values()])
    order = np.argsort(-h)  # coarse -> fine
    monotone = bool(np.all(np.diff(slip[order]) < 0))
    if len(h) >= 3 and slip[order][-1] < slip[order][0]:
        coef = np.polyfit(h, slip, 1)
        slip_extrapolated = float(max(np.polyval(coef, 0.0), 0.0))
    else:
        coef, slip_extrapolated = None, None
    attribution = {
        "monotone_decreasing_with_refinement": monotone,
        "slip_vs_h_fit": None if coef is None else
                        {"slope_per_h": float(coef[0]),
                         "intercept": float(coef[1]),
                         "slip_at_h_zero": slip_extrapolated},
        "decision_rule": ("(a) monotone decrease with refinement => boundary-"
                          "discretization artifact, documented; measured-"
                          "local-shear-rate protocol stands. (b) resolution-"
                          "independent => wall-formulation property; band NOT "
                          "relaxed; formulation-level response required."),
    }
    record = {
        "study_version": "v2 (2026-09-03): kernel co-refined with spacing "
                         "(h = 2*spacing), per the pre-registered decision "
                         "rule; v1 retained as fixed-h control",
        "v1_record": str(V1_BACKUP),
        "v1_control_interpretation": (
            "v1 held h = 1.0 fixed while varying spacing; its scatter "
            "(0.852/0.754/0.814) shows the measured slip is not a particle-"
            "density artifact at fixed kernel, but it could not test h-"
            "convergence. v2 tests h-convergence directly."),
        "solver_state": "post CSF-symmetric-stencil fix + lattice-row profile "
                        "binning (2026-09-02)",
        "protocol": ("exact-row domains; h = 2*spacing and dt = 0.008*h "
                     "(CFL pinned at 0.08, Re = 346 identical at all "
                     "levels); quasi-steady guard (two consecutive "
                     "windows, |dslip| <= 0.005, r2c >= 0.99, base window "
                     "t = 45.6 = the length at which all v1 levels "
                     "measured steady); U_wall = 2; as-delivered "
                     "dissipators (alpha=0.1, xsph=0.1); mu_solvent = 0.05 "
                     "(as-delivered nominal); slip = linear extrapolation "
                     "of the bulk fit to both no-slip planes"),
        "regression_anchor": anchor,
        "configs": rows,
        "steady_guards": guards,
        "attribution": attribution,
    }
    OUT.write_text(json.dumps(record, indent=2))
    print(f"saved -> {OUT}")
    print("REGRESSION ANCHOR:", json.dumps(anchor, indent=1))
    print("ATTRIBUTION:", json.dumps(attribution, indent=1))


if __name__ == "__main__":
    main()
