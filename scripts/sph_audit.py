"""Reproducible SPH audit harness with PRE-REGISTERED gate criteria.

This script is diagnostic: it does not tune parameters or replace any force
formulation. It records the requested observables and evaluates the
zero-shear control against criteria fixed in advance (written here BEFORE
any post-fix full-duration run was inspected):

  GATE (full-duration zero-shear control, sweep config):
    G1  deformation trend  |dD/dt| < 5e-5 per time unit
        (the defective solver drifted at ~1.7e-4/unit: D 0.009 -> 0.078
        over 406 units, monotonic, azimuth locked at ~171 deg)
    G2  bounded oscillation: max |D(t) - D0| < 0.02 over the whole window
        (comparable to the discretization floor D0 ~ 0.016, and ~3.5x
        smaller than the old 0.069 drift)
    G3  COM drift rate < 1e-3 distance units per time unit
    G4  density bounded: rho in [0.98, 1.02] for free particles throughout
    G5  pressure bounded: |p| < 0.5 for free particles throughout
    G6  no NaNs at any sampled step

  Rationale: these thresholds are 3-4x stricter than the observed defect
  and are tied to the physics requirement - a fake drift larger than the
  low-Ca signal (D ~ Ca*O(1), Ca_phys ~ 1e-3..1e-2) would corrupt the
  D-vs-Ca curve. G4/G5 follow from the Laplace records (rho 0.998-1.002,
  |p| < 0.18 in the measurement windows).

Outputs (incremental - a killed run still leaves its trace):
  outputs/sph/audits/zero_shear_<config>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tau_mech.sph import (  # noqa: E402
    SPHParams,
    SPHState,
    build_pairs,
    compute_acceleration,
    compute_density,
    compute_surface_force,
    droplet_deformation,
    make_couette_droplet_state,
    run,
    step,
)

OUT = Path("outputs/sph/audits")

# pre-registered gate thresholds (see module docstring)
GATE = {
    "G1_max_abs_dD_dt": 5e-5,
    "G2_max_abs_D_minus_D0": 0.02,
    "G3_max_com_drift_rate": 1e-3,
    "G4_rho_bounds": [0.98, 1.02],
    "G5_abs_pressure_bound": 0.5,
    "G6_no_nan": 0,
}


def _stats(state):
    droplet = state.phase == 1
    free = state.phase != 2
    d = droplet_deformation(state)
    com = np.asarray(d["com"], dtype=float)
    vel = state.vel[free]
    rho = state.rho[free]
    pressure = state.pressure[free]
    return {
        "deformation": float(d["taylor"]),
        "aspect_ratio": float(d["aspect_ratio"]),
        "angle_deg": float(d["angle_deg"]),
        "com": com.tolist(),
        "rho_mean_free": float(np.mean(rho)),
        "rho_min_free": float(np.min(rho)),
        "rho_max_free": float(np.max(rho)),
        "pressure_mean_free": float(np.mean(pressure)),
        "pressure_min_free": float(np.min(pressure)),
        "pressure_max_free": float(np.max(pressure)),
        "momentum_free": (state.mass[free, None] * vel).sum(axis=0).tolist(),
        "vmax_free": float(np.linalg.norm(vel, axis=1).max()),
        "nan_count": int(np.isnan(state.pos).sum() + np.isnan(state.vel).sum()
                          + np.isnan(state.rho).sum()
                          + np.isnan(state.pressure).sum()),
        "n_droplet": int(droplet.sum()),
    }


def evaluate_gate(trace):
    t = np.array([r["time"] for r in trace])
    D = np.array([r["deformation"] for r in trace])
    D0 = D[0]
    com = np.array([r["com"] for r in trace])
    # guards for short traces (1-2 samples at budget-persist time): a
    # zero-length diff has no max(), and polyfit is degenerate with <2 pts
    com_path = (np.linalg.norm(np.diff(com, axis=0), axis=1)
                if len(com) > 1 else np.array([0.0]))
    slope = float(np.polyfit(t, D, 1)[0]) if len(t) > 1 else 0.0
    checks = {
        "G1_max_abs_dD_dt": {"value": abs(slope),
                              "limit": GATE["G1_max_abs_dD_dt"],
                              "pass": bool(abs(slope) < GATE["G1_max_abs_dD_dt"])},
        "G2_max_abs_D_minus_D0": {"value": float(np.abs(D - D0).max()),
                                    "limit": GATE["G2_max_abs_D_minus_D0"],
                                    "pass": bool(np.abs(D - D0).max()
                                                 < GATE["G2_max_abs_D_minus_D0"])},
        "G3_max_com_drift_rate": {"value": float(com_path.max() / np.diff(t).mean()),
                                    "limit": GATE["G3_max_com_drift_rate"],
                                    "pass": bool(com_path.max() / np.diff(t).mean()
                                                 < GATE["G3_max_com_drift_rate"])},
        "G4_rho_bounds": {"value": [float(min(r["rho_min_free"] for r in trace)),
                                      float(max(r["rho_max_free"] for r in trace))],
                           "limit": GATE["G4_rho_bounds"],
                           "pass": bool(min(r["rho_min_free"] for r in trace)
                                        > GATE["G4_rho_bounds"][0]
                                        and max(r["rho_max_free"] for r in trace)
                                        < GATE["G4_rho_bounds"][1])},
        "G5_abs_pressure_bound": {"value": float(max(
            max(abs(r["pressure_min_free"]), abs(r["pressure_max_free"]))
            for r in trace)),
            "limit": GATE["G5_abs_pressure_bound"],
            "pass": bool(max(max(abs(r["pressure_min_free"]),
                                 abs(r["pressure_max_free"])) for r in trace)
                         < GATE["G5_abs_pressure_bound"])},
        "G6_no_nan": {"value": int(max(r["nan_count"] for r in trace)),
                       "limit": GATE["G6_no_nan"],
                       "pass": bool(max(r["nan_count"] for r in trace) == 0)},
    }
    return {"checks": checks,
             "all_pass": bool(all(c["pass"] for c in checks.values())),
             "D0": float(D0),
             "D_final": float(D[-1]),
             "linear_slope_dD_dt": slope}


def _config_fingerprint(params, steps, dt, every, radius, eq_steps):
    import hashlib
    blob = json.dumps({**params.__dict__, "steps": steps, "dt": dt,
                        "every": every, "radius": radius,
                        "eq_steps": eq_steps}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_zero_shear(name: str, params: SPHParams, steps: int, dt: float,
                   every: int, radius: float = 3.0, eq_steps: int = 0,
                   budget_seconds: float = None):
    """Run (or resume) the zero-shear control; persist state at checkpoints.

    Checkpoint/resume (2026-09-02): background children of the tool wrapper
    are reaped nondeterministically on this machine, so long controls run as
    foreground chunks. Full particle state + trace are persisted to
    state_zero_shear_<name>.npz / zero_shear_<name>.json; a resumed chunk
    continues EXACTLY where the previous one stopped (config fingerprint
    guards against resuming into a changed configuration).
    """
    domain = (0.0, 0.0, 24.0, 16.0)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"zero_shear_{name}.json"
    state_path = OUT / f"state_zero_shear_{name}.npz"
    fp = _config_fingerprint(params, steps, dt, every, radius, eq_steps)

    import time as _time
    t_start = _time.time()

    state = None
    trace = []
    done_to = -1  # last step index whose sample was taken
    if state_path.exists() and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("config_fingerprint") == fp:
                z = np.load(state_path, allow_pickle=False)
                state = SPHState(pos=z["pos"], vel=z["vel"], mass=z["mass"],
                                 phase=z["phase"], h=params.h, rho0=float(z["rho0"]),
                                 domain=domain)
                state.rho = z["rho"]
                state.pressure = z["pressure"]
                trace = prev["trace"]
                done_to = int(z["done_to"])
                print(f"[resume] {name}: continuing from step {done_to} "
                      f"({len(trace)} samples)")
        except Exception as e:
            print(f"[resume] {name}: state unreadable ({e}); starting fresh")
            state = None

    if state is None:
        state = make_couette_droplet_state(
            params, domain=domain, spacing=0.5, droplet_radius=radius,
            n_wall_layers=4)
        run(state, params, 0, dt)
        if eq_steps > 0:
            # equilibrate exactly like the sweep (walls at rest) so the gate
            # measures the SAME window the drift polluted: post-equilibration.
            run(state, params, eq_steps, dt)
        trace = []
        done_to = -1

    def persist():
        np.savez(state_path, pos=state.pos, vel=state.vel, mass=state.mass,
                 phase=state.phase, rho=state.rho, pressure=state.pressure,
                 rho0=state.rho0, done_to=done_to)
        result = _write(out_path, name, params, steps, dt, every, radius,
                        trace, eq_steps)
        result["config_fingerprint"] = fp
        out_path.write_text(json.dumps(result, indent=2))
        return result

    # if the very first sample (step 0) has not been taken yet, take it
    if done_to < 0:
        trace.append({"step": 0, "time": 0.0, **_stats(state)})
        done_to = 0

    while done_to < steps:
        s = done_to + 1
        step(state, params, dt)
        if s % every == 0:
            trace.append({"step": s, "time": s * dt, **_stats(state)})
        done_to = s
        if s % (every * 20) == 0:
            persist()
        if budget_seconds is not None \
                and (_time.time() - t_start) > budget_seconds:
            result = persist()
            print(f"[budget] {name}: paused at step {done_to}/{steps} "
                  f"({len(trace)} samples); resume with the same command")
            return result
    result = persist()
    return result


def _write(out_path, name, params, steps, dt, every, radius, trace, eq_steps=0):
    result = {
        "name": name,
        "params": params.__dict__,
        "domain": [0.0, 0.0, 24.0, 16.0],
        "radius": radius,
        "eq_steps": eq_steps,
        "steps": steps,
        "dt": dt,
        "trace_every": every,
        "trace": trace,
        "gate": evaluate_gate(trace),
        "gate_criteria_version": "2026-09-02 (pre-registered before the "
                                  "post-fix full-duration run)",
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


def csf_symmetry_audit():
    p = SPHParams()
    state = make_couette_droplet_state(
        p, domain=(0.0, 0.0, 24.0, 16.0), spacing=0.5,
        droplet_radius=3.0, n_wall_layers=4)
    run(state, p, 0, 0.008)
    pairs, d, e = build_pairs(state.pos, p.h,
                              x_period=state.domain[2] - state.domain[0])
    compute_density(state, p, pairs, d)
    pressure_force = compute_acceleration(
        state, replace(p, sigma_surf=0.0, A_surf=0.0, B_surf=0.0,
                       mu_solvent=0.0, mu_droplet=0.0, alpha_art=0.0),
        pairs, d, e)
    surface_force = compute_surface_force(state, p, pairs, d, e)
    free = state.phase != 2
    return {
        "csf_force_sum_mass_weighted": ((state.mass[free, None]
                                          * surface_force[free]).sum(axis=0)).tolist(),
        "pressure_force_sum_mass_weighted": ((state.mass[free, None]
                                               * pressure_force[free]).sum(axis=0)).tolist(),
        "note": "both are INTERNAL interactions; mass-weighted sums must "
                "vanish to discretization error (see "
                "scripts/diag_csf_symmetry.py for the operator-level audit)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dt", type=float, default=0.008)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--mu-solvent", type=float, default=None,
                    help="override mu_solvent (default: SPHParams())")
    ap.add_argument("--mu-droplet", type=float, default=None,
                    help="override mu_droplet (default: SPHParams())")
    ap.add_argument("--configs", default="baseline",
                    help="comma-separated subset: baseline,no_csf,no_xsph,"
                         "no_artificial_viscosity,no_shepard,no_immiscibility")
    ap.add_argument("--eq-steps", type=int, default=0,
                    help="equilibration steps before the gate window "
                         "(the sweep uses 4000)")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="run at most this long, persist, and exit cleanly "
                         "(for chunked foreground execution)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base = SPHParams()
    if args.mu_solvent is not None:
        base.mu_solvent = args.mu_solvent
    if args.mu_droplet is not None:
        base.mu_droplet = args.mu_droplet
    configs = {
        "baseline": base,
        "no_csf": replace(base, sigma_surf=0.0),
        "no_xsph": replace(base, xsph=0.0),
        "no_artificial_viscosity": replace(base, alpha_art=0.0),
        "no_shepard": replace(base, shepard=0.0),
        "no_immiscibility": replace(base, A_surf=0.0, B_surf=0.0),
        # smoke-test alias of baseline (identical physics) used to validate
        # the checkpoint/resume path itself
        "resume_test": base,
    }
    for name in args.configs.split(","):
        r = run_zero_shear(name, configs[name], args.steps, args.dt, args.every,
                           eq_steps=args.eq_steps,
                           budget_seconds=args.budget_seconds)
        g = r["gate"]
        print(f"{name}: D0={g['D0']:.4f} D_final={g['D_final']:.4f} "
              f"slope={g['linear_slope_dD_dt']:.2e} all_pass={g['all_pass']}")
        for k, c in g["checks"].items():
            print(f"   {k}: value={c['value']} limit={c['limit']} pass={c['pass']}")


if __name__ == "__main__":
    main()
