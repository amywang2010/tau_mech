"""Phase 3 - droplet-in-shear study entry point.

Equilibrates a Tau-condensate-like droplet (2D WCSPH prototype) at rest in a
Couette cell, then applies shear at several rates and records deformation
descriptors (Taylor deformation D, aspect ratio, orientation angle) over
time. The capillary number Ca = mu_droplet * gamma_dot * R / sigma is the
physically meaningful reporting quantity.

Physiological mapping (documented in README, phase 3): CSF/perivascular
shear stresses tau ~ 0.1-1 Pa on a R ~ 1 um droplet with mu_droplet ~ 1e2
Pa s and sigma ~ 1e-4 N/m give Ca = tau*R/sigma ~ 1e-3..1e-2. The sweep
spans a wider Ca range to map the full response curve; the physiological
point is reported separately with its expected deformation.

Usage:
    python scripts/sph_sweep.py [--shear-rates 0,0.02,0.05,0.1,0.2]
                                [--eq-steps N] [--shear-steps N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams, droplet_shear_sweep, validate_couette, validate_laplace,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shear-rates", type=str,
                   default="0.0,0.001,0.003,0.01,0.03,0.1")
    p.add_argument("--eq-steps", type=int, default=4000)
    p.add_argument("--shear-steps", type=int, default=60000)
    p.add_argument("--dt", type=float, default=0.008)
    p.add_argument("--mu-solvent", type=float, default=1.0)
    p.add_argument("--mu-droplet", type=float, default=10.0)
    p.add_argument("--couette-steps", type=int, default=0,
                   help="0 = skip the (slow) Couette validation")
    p.add_argument("--laplace-steps", type=int, default=2500)
    args = p.parse_args()

    # ELEVATED viscosities (lambda = mu_d/mu_s = 10 preserved) so the Couette
    # flow develops and the droplet relaxes within a feasible run; the flow
    # development time t_flow = H_wall^2/nu and the capillary time
    # t_char = mu_d*R/sigma are both reported per case, and the sweep uses
    # the MEASURED local shear rate for Ca (see droplet_shear_sweep).
    params = SPHParams(mu_solvent=args.mu_solvent,
                       mu_droplet=args.mu_droplet)
    out_dir = os.path.join("outputs", "sph")
    os.makedirs(out_dir, exist_ok=True)
    shear_rates = [float(x) for x in args.shear_rates.split(",")]
    print(f"Sweep viscosities: mu_solvent={params.mu_solvent} "
          f"mu_droplet={params.mu_droplet} (lambda="
          f"{params.mu_droplet / params.mu_solvent})")

    # ---- validation block (fast parts) -----------------------------------
    if args.couette_steps > 0:
        print("=== Couette (steady-state linear profile, nu=0.5, H=8) ===")
        c = validate_couette(params, n_steps=args.couette_steps, dt=args.dt,
                             domain=(0.0, 0.0, 24.0, 8.0))
        print(f"  R2_fit={c['r2_fit']:.4f} R2_analytic={c['r2_analytic']:.4f} "
              f"slope_ratio={c['slope_ratio']:.4f} t_sim={c['t_sim']:.2f}")
    else:
        c = None

    print("=== Laplace calibration ===")
    # reuse the authoritative record from scripts/diag_surface_tension.py
    # when present (identical measurement); fall back to a fresh run
    calib_path = os.path.join(out_dir, "laplace_calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            lap = json.load(f)
        print(f"  [reused {calib_path}]")
        # the persisted record and the raw validate_laplace dict have
        # different schemas; normalize both (review fix)
        if "per_radius" in lap:
            rows = {float(R): v for R, v in lap["per_radius"].items()}
            # prefer the mean per-radius sigma_eff (the physically effective
            # tension; ~0.97 sigma_input after the factor-2 fix) over the 1/R
            # slope fit
            sigma = float(lap.get("sigma_eff", lap.get("sigma_fit",
                                                        float("nan"))))
            lin = float(lap.get("linearity_dP_vs_1R", float("nan")))
            usable = bool(lap.get("usable", False))
        else:
            rows = {float(k): v for k, v in lap.items()
                    if not k.startswith("_")}
            sigma = float(lap["_sigma_fit"])
            lin = float(lap["_linearity"])
            usable = np.isfinite(sigma) and abs(lin) > 0.8
    else:
        lap = validate_laplace(params, n_steps=args.laplace_steps, dt=args.dt)
        rows = {float(k): v for k, v in lap.items() if not k.startswith("_")}
        sigma = float(lap["_sigma_fit"])
        lin = float(lap["_linearity"])
        usable = np.isfinite(sigma) and abs(lin) > 0.8
    for R in sorted(rows):
        print(f"  R={R}: dP={rows[R]['dP']:.4f}")
    print(f"  sigma_eff={sigma:.4f}  linearity={lin:.4f}  usable={usable}")
    if not (np.isfinite(sigma) and sigma > 0.0):
        print("  ERROR: no usable surface tension (sigma <= 0 or non-finite); "
              "run scripts/diag_surface_tension.py first and check its "
              "'usable' flag before running the sweep")
        sys.exit(1)
    if not usable:
        print("  WARNING: the reused calibration record is marked NOT USABLE; "
              "the sweep proceeds with sigma_fit but the Laplace scaling was "
              "not verified - check scripts/diag_surface_tension.py output")
    sigma = float(sigma)

    # ---- shear sweep -------------------------------------------------------
    print(f"=== Droplet-in-shear sweep: gamma_dot = {shear_rates} ===")
    rows = droplet_shear_sweep(
        params, shear_rates=shear_rates, eq_steps=args.eq_steps,
        shear_steps=args.shear_steps, dt=args.dt, sigma=sigma,
        out_dir=out_dir)

    # ---- figure: EXCESS deformation Delta D = D - D0 vs Ca ------------------
    # The discrete droplet has an intrinsic roundness floor D0 ~ 0.016 (the
    # inertia tensor of a particle circle is not exactly a circle); the shear
    # response is the EXCESS deformation above that settled baseline, which is
    # the quantity the Taylor (1934) small-Ca law predicts (D=0 at Ca=0). The
    # baseline is taken from the no-shear control (gamma_dot=0) when present,
    # else the mean initial D across cases.
    ca = [r["capillary_number_Ca"] for r in rows]
    d0 = [r["taylor_initial"] for r in rows]
    dplateau = [r["taylor_plateau_fit"] for r in rows]
    dfinal = [r["taylor_final"] for r in rows]
    ca_nom = [r["capillary_number_nominal"] for r in rows]
    zero = [i for i, r in enumerate(rows) if r["shear_rate_nominal"] == 0.0]
    D0 = float(np.mean([d0[i] for i in zero])) if zero else float(np.mean(d0))
    dD_plateau = [p - D0 for p in dplateau]
    dD_final = [f - D0 for f in dfinal]
    dD0 = [i - D0 for i in d0]
    lam = params.mu_droplet / params.mu_solvent
    taylor_coef = (19.0 * lam + 16.0) / (16.0 * lam + 16.0)
    ca_th = np.logspace(np.log10(max(min(ca_nom) * 0.5, 1e-4)),
                        np.log10(max(ca_nom) * 2.0), 40)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ca, dD0, "s--", color="0.6", label="initial (settled) − D$_0$")
    ax.plot(ca, dD_plateau, "o-", color="crimson",
            label="plateau ΔD$_\\infty$ (fit) − D$_0$")
    ax.plot(ca_nom, dD_final, "^:", color="0.4",
            label="final sample − D$_0$")
    ax.plot(ca_th, taylor_coef * np.asarray(ca_th), "--", color="steelblue",
            label=f"Taylor (1934) small-Ca limit (a={taylor_coef:.2f})")
    ax.set_xlabel("capillary number Ca (measured $\\mu_d\\dot{\\gamma} R/\\sigma$)")
    ax.set_ylabel("excess Taylor deformation ΔD = D − D$_0$")
    ax.set_xscale("log")
    ax.set_title(f"Tau droplet deformation vs shear (2D WCSPH; D$_0$={D0:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sph_deformation_vs_Ca.png"), dpi=150)
    plt.close(fig)
    print(f"\n  [sph] settled baseline D0 = {D0:.4f} (no-shear control)")

    # ---- physiological point ------------------------------------------------
    # tau = 0.1-1 Pa, R = 1 um, sigma = 1e-4 N/m  =>  Ca = tau*R/sigma ~ 1e-3..1e-2
    # The CPU prototype cannot simulate Ca < ~0.1 in feasible runtime (the
    # deformation timescale 1/gamma_dot would require ~10^5-10^6 steps), so the
    # physiological deformation is reported from the ANALYTIC Taylor (1934)
    # small-deformation limit D = Ca * (19*lam + 16) / (16*lam + 16), which is
    # valid for Ca << 1, rather than by extrapolating the finite-Ca sweep.
    lam = params.mu_droplet / params.mu_solvent
    taylor_coef = (19.0 * lam + 16.0) / (16.0 * lam + 16.0)
    print("\nPhysiological extrapolation (R=1 um, sigma=1e-4 N/m, Taylor limit):")
    phys = {}
    for tau_pa in (0.1, 1.0):
        ca_phys = tau_pa * 1e-6 / 1e-4  # Ca = tau*R/sigma (R=1e-6 m)
        d_pred = ca_phys * taylor_coef
        phys[f"D_at_{tau_pa}Pa"] = d_pred
        phys[f"Ca_at_{tau_pa}Pa"] = ca_phys
        print(f"  tau={tau_pa} Pa -> Ca={ca_phys:.4f} -> D ~ {d_pred:.4f} "
              f"(Taylor limit; Ca << 1)")

    summary = {
        "validation": {"couette": c, "laplace": lap},
        "settled_baseline_D0": D0,
        "equilibration": {
            "eq_steps": args.eq_steps,
            "note": ("droplet equilibrated at rest (walls stationary) for "
                     "eq_steps before shear; the initial transient is small "
                     "because the periodic-x neighbour search and the wall "
                     "lattice are correct (no artificial drag used).")},
        "sweep_rows": rows,
        "physiology": {
            "note": ("Ca = tau*R/sigma with tau=0.1-1 Pa, R=1e-6 m, "
                     "sigma=1e-4 N/m gives Ca ~ 1e-3..1e-2. Simulated sweep "
                     "covers Ca >= ~0.1; physiological deformation uses the "
                     "analytic Taylor (1934) small-deformation limit "
                     "D = Ca*(19*lam+16)/(16*lam+16) with lam = mu_d/mu_s."),
            "lambda_viscosity_ratio": lam,
            **phys,
        },
    }
    with open(os.path.join(out_dir, "sph_study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
