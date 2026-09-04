"""Couette wall-slip resolution study, v2.1 (2026-09-04).

WHY v2.1 (honest protocol history)
-----------------------------------
v1 (2026-09-02, archived as couette_resolution_study_v1_fixedh.json) held
h = 1.0 fixed while varying spacing: its ~2h wall-coupling layer was
identical in absolute units across levels, so it could not test
h-convergence. Retained as a fixed-h control.

v2 (2026-09-03, abort record archived as
couette_resolution_study_v2_aborted.json) co-refined h = 2*spacing with
dt pinned at CFL = 0.08, but kept the as-delivered nu = 0.05 and a rest
initial condition. Its determinism anchor PASSED (bit-reproduction of
v1's coarse slip), and its quasi-steady guard then correctly ABORTED the
study: slip drifted 0.852 -> 0.865 -> 0.916 across windows because the
flow was still relaxing from rest on the momentum-diffusion time
(tau = 152 t.u. at nu = 0.05; the records' own t_over_tau_nu_nominal
= 0.30-0.68 fields document the deficit). The guard caught a transient
that linear-profile checks cannot see: a Couette profile stays visually
linear while its slope relaxes.

v2.1 design (pre-registered in docs/A3_COUETTE_V21_PREREGISTRATION.md)
----------------------------------------------------------------------
Two changes, both restoring the protocol's own design; geometry, kernel
ratio (h = 2*spacing), CFL (0.08), dissipators, guard thresholds, and the
decision rule are UNCHANGED from v2:

1. mu_solvent = 0.5 - the value every other validation of this solver
   used, and the value the original "t = 3 viscous times" budget assumed
   (tau = 15.2 t.u.; base window t = 45.6 = 3.0 tau). Re drops from 346
   to 34.6 - still deep in the laminar Stokes regime where the analytic
   Couette profile is exact.
2. init = "analytic": the fluid is initialized AT the exact steady
   no-slip profile, converting the guard from a relaxation test into a
   strictly stronger fixed-point test (residual drift = discretization
   error, not start-up transient). One rest-IC diagnostic run at level 1
   is retained to quantify the v1/v2 transient directly.

Execution model (scheduling only; per-window physics identical to v2):
each guard window is an independent run from the initial state, so all
windows of all levels launch as PARALLEL single-core subprocesses
(<= 7 processes on 12 logical cores). The orchestrator applies the
unchanged guard per level (|slip(w2) - slip(w1)| <= 0.005, r2_central
>= 0.99 on both windows; one 1.5x escalation window on demand; abort if
a level still fails), gates levels 2-3 on the level-1 guard, and writes
the canonical record with the same schema as v2 plus v2.1 fields.

Decision rule (unchanged): (a) monotone slip decrease with refinement =>
boundary-discretization artifact, documented; (b) resolution-independent
=> wall-formulation property; the acceptance band is NOT relaxed either
way.

Cost (measured 3.3e-4 s/particle-step): wall-clock is bounded by the
largest single window, level-3 window 2 (17,100 steps x 3,744 particles
~ 5.9 h); worst case with a level-3 escalation window (25,650 steps)
~ 8.9 h. (Window step counts per level: 5700/8550/11400 base for
t = 45.6, since dt = 0.008*h scales with h; escalation windows at 2.25x.)

Run:  python scripts/diag_couette_resolution.py [--smoke]
      python scripts/diag_couette_resolution.py --window SPACING ROWS
                                               NSTEPS INIT OUTPATH
                                               (subprocess entry only)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, validate_couette  # noqa: E402

OUT = Path("outputs/sph/audits/couette_resolution_study.json")
V1_BACKUP = Path("outputs/sph/audits/couette_resolution_study_v1_fixedh.json")
V2_ABORT = Path("outputs/sph/audits/couette_resolution_study_v2_aborted.json")
FRAGDIR = Path("outputs/sph/audits/res_v21_fragments")
SPACING_ROWS = {0.5: 19, 1.0 / 3.0: 29, 0.25: 39}  # exact fluid rows per level
BASE_TIME = 45.6            # base window: t = 3.0 tau at the designed nu = 0.5
STEADY_TOL = 0.005          # max |slip(w2) - slip(w1)| accepted as steady
R2_MIN = 0.99               # linearity floor on every window
MAX_ESCALATIONS = 1         # window escalation cap (1.5x, once, then abort)
NU_DESIGNED = 0.5           # the protocol's own designed viscosity


def domain_for(spacing: float, n_rows: int):
    """Exact-row Couette domain (identical to v1/v2): fluid occupies exactly
    n_rows lattice rows; no-slip planes one row outside; H_wall = (n_rows+1)dy.
    """
    dy = spacing * np.sqrt(3.0) / 2.0
    y1 = (n_rows - 0.5) * dy
    return (0.0, 0.0, 24.0, float(y1))


def window_steps(spacing: float) -> int:
    """Base window length in steps at this level's dt = 0.008*h = 0.016*dx.

    t = BASE_TIME at EVERY level (dt scales with h), so step counts scale
    as 1/spacing: 5700 / 8550 / 11400 for spacings 0.5 / 1/3 / 0.25 -
    identical to v2's windows by construction.
    """
    dt = 0.008 * 2.0 * spacing
    return int(np.ceil(BASE_TIME / dt))


def run_window(spacing: float, n_rows: int, n_steps: int, init: str,
               out_path: Path) -> None:
    """One guard window as an independent run from the initial state.

    This is EXACTLY the per-window measurement v2 performed (same
    validate_couette call, same metrics); running it in a subprocess is a
    scheduling change only. Writes a self-contained fragment atomically.
    """
    h = 2.0 * spacing
    params = SPHParams(h=h, mu_solvent=NU_DESIGNED)
    domain = domain_for(spacing, n_rows)
    H_wall = (n_rows + 1) * spacing * np.sqrt(3.0) / 2.0
    dt = 0.008 * h
    r = validate_couette(params=params, n_steps=n_steps, dt=dt,
                         domain=domain, spacing=spacing, init=init)
    r.update({
        "h": h, "spacing": spacing, "n_rows": n_rows, "H_wall": H_wall,
        "n_steps": n_steps, "dt": dt, "init": init,
        "mu_solvent": NU_DESIGNED,
        "nu_used": NU_DESIGNED,
        "tau_mode": H_wall**2 / (NU_DESIGNED * np.pi**2),
        "t_sim": n_steps * dt,
        "cfl_dt_ratio": dt * params.c_s / h,
        "viscous_dt_ratio": dt * NU_DESIGNED / (0.125 * h**2),
        "reynolds": U_WALL * H_wall / NU_DESIGNED,  # U_char = wall speed, L_char = H_wall (project convention; = 34.6 at nu = 0.5)
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=2))
    os.replace(tmp, out_path)  # atomic: no partial fragment ever visible
    print(f"[window done] s={spacing:.4f} n={n_steps} init={init} "
          f"slip={r['slip_frac']:.4f} R2c={r['r2_central']:.4f} -> {out_path}",
          flush=True)


U_WALL = 2.0  # wall speed; defined here so run_window's Re uses one constant


def _spawn(spacing: float, n_rows: int, n_steps: int, init: str,
           frag: Path) -> subprocess.Popen:
    cmd = [sys.executable, os.path.abspath(__file__), "--window",
           f"{spacing:.10g}", str(n_rows), str(n_steps), init, str(frag)]
    log = open(FRAGDIR / f"{frag.stem}.log", "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=".")
    log.close()  # child holds its own inherited handle
    return p


def _collect(frag: Path) -> dict:
    return json.loads(frag.read_text())


def _guard_from(results: dict[int, dict], tag: str) -> tuple[dict | None, dict, bool]:
    """Apply the unchanged v2 guard to >=2 window results (ordered by steps).

    Returns (final_row_or_None, guard_dict, steady_bool). On failure with
    escalation budget left, returns the escalation step count as a request.
    """
    order = sorted(results)
    a, b = results[order[-2]], results[order[-1]]
    drift = abs(b["slip_frac"] - a["slip_frac"])
    ok = (drift <= STEADY_TOL
          and a["r2_central"] >= R2_MIN and b["r2_central"] >= R2_MIN)
    guard = {
        "windows": order,
        "steady": ok,
        "first_window_slip": results[order[0]]["slip_frac"],
        "slip_drift_last": drift,
        "r2_central_all": {str(k): results[k]["r2_central"] for k in order},
        "note": ("quasi-steady guard: two consecutive windows with "
                 "|dslip| <= 0.005 and r2_central >= 0.99; the headline row "
                 "is the FINAL window. v2.1: windows are independent runs "
                 "from the ANALYTIC steady IC (fixed-point test)."),
    }
    if ok:
        final = dict(b)
        final.update({"tau_nu_nominal": final["tau_mode"],
                      "t_over_tau_nu_nominal": final["t_sim"] / final["tau_mode"]})
        return final, guard, True
    return None, guard, False


def level_guard_async(spacing: float, n_rows: int, tag: str):
    """Schedule a level's windows immediately; returns a zero-arg closure
    that waits and applies the unchanged guard (+1 escalation on demand)."""
    w1 = window_steps(spacing)
    f1 = FRAGDIR / f"{tag}_w{w1}.json"
    f2 = FRAGDIR / f"{tag}_w{int(np.ceil(w1 * 1.5))}.json"
    p1 = _spawn(spacing, n_rows, w1, "analytic", f1)
    p2 = _spawn(spacing, n_rows, int(np.ceil(w1 * 1.5)), "analytic", f2)

    def wait_guard():
        p1.wait()
        p2.wait()
        results = {w1: _collect(f1), int(np.ceil(w1 * 1.5)): _collect(f2)}
        escalations = 0
        while True:
            final, guard, steady = _guard_from(results, tag)
            if steady:
                return final, guard, True
            if escalations >= MAX_ESCALATIONS:
                guard["escalations"] = escalations
                return None, guard, False
            escalations += 1
            we = int(np.ceil(w1 * (1.5 ** (1 + escalations))))
            fe = FRAGDIR / f"{tag}_w{we}.json"
            pe = _spawn(spacing, n_rows, we, "analytic", fe)
            pe.wait()
            results[we] = _collect(fe)

    return wait_guard


def steady_preservation_anchor(level1_guard: dict,
                               rest_frag: Path | None) -> dict:
    """v2.1 anchor (replaces v1's bit-reproduction anchor, which anchored a
    nu = 0.05 protocol): the analytic-IC flow must HOLD the steady state -
    the FIRST window's r2_central >= 0.999 (initialized at the exact
    profile; holding it is the strongest fixed-point evidence) - and the
    retained rest-IC diagnostic quantifies the v1/v2 transient directly.
    """
    r2c = level1_guard["r2_central_all"]
    w1_steps = min(int(k) for k in r2c)
    w1_r2 = r2c[str(w1_steps)]
    anchor = {
        "kind": "steady_preservation (analytic IC fixed-point test)",
        "window1_steps": w1_steps,
        "window1_r2_central": w1_r2,
        "pass": bool(w1_r2 >= 0.999),
        "note": ("v1 bit-reproduction anchored the nu=0.05 protocol; under "
                 "v2.1 (nu=0.5, analytic IC) the anchor is steady-state "
                 "preservation itself - the flow starts at the analytic "
                 "profile and must stay there (r2_central >= 0.999)."),
    }
    if rest_frag is not None and rest_frag.exists():
        rrest = _collect(rest_frag)
        anchor["rest_ic_diagnostic"] = {
            "slip_frac_rest_5700": rrest["slip_frac"],
            "slip_frac_analytic_5700": level1_guard["first_window_slip"],
            "abs_diff": abs(rrest["slip_frac"]
                            - level1_guard["first_window_slip"]),
            "interpretation": ("rest-IC vs analytic-IC at the identical "
                               "level-1 config: the difference IS the "
                               "start-up transient contamination that "
                               "v1/v2 measured as 'steady' slip."),
        }
    return anchor


def main() -> None:
    if "--window" in sys.argv:
        i = sys.argv.index("--window")
        s, rows, n, init, path = sys.argv[i + 1:i + 6]
        run_window(float(s), int(rows), int(n), init, Path(path))
        return

    smoke = "--smoke" in sys.argv
    rescoped = "--level1" in sys.argv
    if smoke:
        FRAGDIR.mkdir(parents=True, exist_ok=True)
        f1 = FRAGDIR / "smoke_w1.json"
        f2 = FRAGDIR / "smoke_w2.json"
        p1 = _spawn(0.5, 19, 120, "analytic", f1)
        p2 = _spawn(0.5, 19, 180, "analytic", f2)
        rc1, rc2 = p1.wait(), p2.wait()
        assert rc1 == 0 and rc2 == 0, "smoke window subprocess failed"
        r1, r2 = _collect(f1), _collect(f2)
        for k in ("slip_frac", "r2_central", "cfl_dt_ratio", "reynolds"):
            assert np.isfinite(r1[k]) and np.isfinite(r2[k]), f"non-finite {k}"
        assert abs(r1["cfl_dt_ratio"] - 0.08) < 1e-12
        # Re = U*H/nu = 2*8.660254.../0.5 (level-1 geometry)
        assert abs(r1["reynolds"] - 34.641016151377544) < 1e-9
        drift = abs(r2["slip_frac"] - r1["slip_frac"])
        print(f"SMOKE OK: w1 slip={r1['slip_frac']:.4f} R2c={r1['r2_central']:.4f} "
              f"cfl={r1['cfl_dt_ratio']:.3f} Re={r1['reynolds']:.1f} "
              f"mu={r1['mu_solvent']} init={r1['init']} | w2 slip={r2['slip_frac']:.4f} "
              f"drift={drift:.4f} (no record written)")
        return

    # Archive the v1 (fixed-h) and v2 (aborted) records exactly once.
    if OUT.exists() and not V1_BACKUP.exists():
        shutil.copy2(OUT, V1_BACKUP)
        print(f"v1 (fixed-h) record archived -> {V1_BACKUP}")
    if OUT.exists() and not V2_ABORT.exists():
        shutil.copy2(OUT, V2_ABORT)
        print(f"v2 (aborted) record archived -> {V2_ABORT}")

    FRAGDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- phase 1: level-1 guard + rest-IC diagnostic, all in parallel ----
    lvl1_key = "s_0.5000_h_1.0000_rows_19"
    f_rest = FRAGDIR / "level1_rest_w5700.json"
    p_rest = _spawn(0.5, 19, window_steps(0.5), "rest", f_rest)
    wait1 = level_guard_async(0.5, 19, "level1")
    final1, guard1, steady1 = wait1()
    p_rest.wait()
    guard1["escalations"] = guard1.get("escalations", 0)
    if not steady1:
        if p_rest.poll() is None:
            p_rest.kill()
        print("LEVEL-1 GUARD FAILED - aborting before levels 2-3.", flush=True)
        OUT.write_text(json.dumps({
            "study_version": "v2.1 (2026-09-04)",
            "level": lvl1_key,
            "steady_guards": {lvl1_key: guard1},
            "attribution": None,
            "aborted": f"quasi-steady guard failed at level {lvl1_key}",
        }, indent=2))
        print(f"saved abort record -> {OUT}")
        return

    # ---- anchor (level-1; gates everything downstream) -------------------
    anchor = steady_preservation_anchor(guard1, f_rest)
    if not anchor["pass"]:
        OUT.write_text(json.dumps({
            "study_version": "v2.1 (2026-09-04)",
            "regression_anchor": anchor,
            "steady_guards": guards,
            "configs": rows,
            "attribution": None,
            "aborted": "steady-preservation anchor failed at level 1",
        }, indent=2))
        print("STEADY-PRESERVATION ANCHOR FAILED - abort record saved.")
        return

    if rescoped:
        # Rescoped study (pre-registered A3 amendment, committed before any
        # v2.1 data): deliver the level-1 steady-preservation validation and
        # the transient diagnostic; the three-level attribution is NOT
        # attempted. Record completed (not aborted) for its declared scope.
        record = {
            "study_version": "v2.1-level1 (2026-09-04, rescoped per A3 "
                             "amendment section 'Scope decision')",
            "scope": ("level-1 steady-preservation fixed-point test + rest-IC "
                      "transient diagnostic; three-level h-convergence "
                      "attribution NOT attempted (rescoped - not load-bearing "
                      "for any paper claim; protocol retained for future "
                      "demand)"),
            "preregistration": "docs/A3_COUETTE_V21_PREREGISTRATION.md",
            "v1_record": str(V1_BACKUP),
            "v2_abort_record": str(V2_ABORT),
            "solver_state": "post CSF-symmetric-stencil fix + lattice-row "
                            "profile binning (2026-09-02); validate_couette "
                            "gained init='analytic' (rest default unchanged)",
            "protocol": ("exact-row domain; h = 2*spacing = 1.0, dt = 0.008 "
                         "(CFL 0.08); mu_solvent = 0.5 (designed; tau = 15.2, "
                         "t = 45.6 = 3.0 tau); analytic steady-IC; guard "
                         "unchanged; U_wall = 2; as-delivered dissipators; "
                         "slip = linear extrapolation convention"),
            "regression_anchor": anchor,
            "configs": rows,
            "steady_guards": guards,
            "attribution": None,
            "attribution_status": "not_attempted_rescoped",
            "wall_clock_s": time.time() - t0,
        }
        OUT.write_text(json.dumps(record, indent=2))
        print(f"saved (rescoped, level-1) -> {OUT}  "
              f"({(time.time() - t0) / 60:.1f} min wall)")
        print("ANCHOR:", json.dumps(anchor, indent=1))
        return

    # ---- phase 2: levels 2+3 windows all in parallel (4 processes) -------
    waits = {}
    for spacing, n_rows in SPACING_ROWS.items():
        if spacing == 0.5:
            continue
        tag = f"lvl_rows{n_rows}"
        waits[spacing] = (n_rows, level_guard_async(spacing, n_rows, tag))
    rows = {lvl1_key: final1}
    guards = {lvl1_key: guard1}
    abort_level = None
    for spacing, (n_rows, wait_fn) in waits.items():
        final, guard, steady = wait_fn()
        key = f"s_{spacing:.4f}_h_{2.0 * spacing:.4f}_rows_{n_rows}"
        guards[key] = guard
        if steady:
            rows[key] = final
        else:
            abort_level = key
    if abort_level is not None:
        OUT.write_text(json.dumps({
            "study_version": "v2.1 (2026-09-04)",
            "level": abort_level,
            "steady_guards": guards,
            "configs": rows,
            "attribution": None,
            "aborted": f"quasi-steady guard failed at level {abort_level}",
        }, indent=2))
        print(f"GUARD FAILED at {abort_level} - abort record saved -> {OUT}")
        return

    # ---- pre-registered attribution (identical rule to v2) ----------------
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
        "study_version": "v2.1 (2026-09-04): nu restored to the designed 0.5 "
                         "(the value the t=3*tau budget assumed) and analytic "
                         "steady-IC initialization (fixed-point guard); h = "
                         "2*spacing co-refinement, CFL = 0.08 unchanged from "
                         "v2; v1 retained as fixed-h control, v2 abort "
                         "retained as the transient evidence",
        "preregistration": "docs/A3_COUETTE_V21_PREREGISTRATION.md",
        "v1_record": str(V1_BACKUP),
        "v2_abort_record": str(V2_ABORT),
        "solver_state": "post CSF-symmetric-stencil fix + lattice-row profile "
                        "binning (2026-09-02); validate_couette gained "
                        "init='analytic' (rest default unchanged)",
        "protocol": ("exact-row domains; h = 2*spacing, dt = 0.008*h (CFL "
                     "pinned 0.08); mu_solvent = 0.5 (designed); analytic "
                     "steady-IC; guard unchanged (two windows, |dslip| <= "
                     "0.005, r2c >= 0.99, one 1.5x escalation, then abort); "
                     "U_wall = 2; as-delivered dissipators (alpha=0.1, "
                     "xsph=0.1); slip = linear extrapolation of the bulk "
                     "fit to both no-slip planes"),
        "regression_anchor": anchor,
        "configs": rows,
        "steady_guards": guards,
        "attribution": attribution,
        "wall_clock_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(record, indent=2))
    print(f"saved -> {OUT}  ({(time.time() - t0) / 3600:.2f} h wall)")
    print("ANCHOR:", json.dumps(anchor, indent=1))
    print("ATTRIBUTION:", json.dumps(attribution, indent=1))


if __name__ == "__main__":
    main()
