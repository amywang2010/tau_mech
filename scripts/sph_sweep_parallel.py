"""Parallel Phase-3 shear sweep driver.

Each shear rate is an INDEPENDENT simulation (a fresh droplet is built and
equilibrated per rate; the rates share no mutable state), so the 6 rates can
run concurrently on separate cores. On the single-threaded numpy WCSPH engine
(~0.34 s/step) a sequential 6-rate sweep is ~31 h of CPU; running the rates in
parallel cuts wall-clock ~6x with no change to the physics or per-step code.

Design
------
* Driver mode (default) reads the Laplace calibration, then launches one
  worker process per unfinished rate. Each worker writes its own per-rate
  JSON + traces under outputs/sph/workers/<rate>/.
* Worker mode (--worker --rate X) runs a SINGLE rate via
  sph.droplet_shear_sweep into a per-rate directory. Because
  droplet_shear_sweep already checkpoints per rate, a crashed worker can be
  re-launched and it will resume from its own directory.
* After all workers finish, the driver merges the per-rate rows (sorted by
  shear rate), writes the canonical sph_shear_sweep.json + sph_traces.npz,
  regenerates the D-vs-Ca figure, and writes sph_study_summary.json.

Usage:
    python scripts/sph_sweep_parallel.py            # all 6 rates, parallel
    python scripts/sph_sweep_parallel.py --max-workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, droplet_shear_sweep  # noqa: E402

OUT = os.path.join("outputs", "sph")
WORKERS = os.path.join(OUT, "workers")
DEFAULT_RATES = "0.0,0.001,0.003,0.01,0.03,0.1"


def load_sigma() -> float:
    calib_path = os.path.join(OUT, "laplace_calibration.json")
    with open(calib_path) as f:
        lap = json.load(f)
    sigma = float(lap.get("sigma_eff", lap.get("sigma_fit", float("nan"))))
    if not np.isfinite(sigma) or sigma <= 0:
        raise SystemExit(f"no usable sigma in {calib_path}; run "
                         "scripts/diag_surface_tension.py first")
    return sigma


def worker(rate: float, sigma: float, args) -> None:
    """Run a single rate into its per-rate directory (idempotent / resumable)."""
    params = SPHParams(mu_solvent=args.mu_solvent, mu_droplet=args.mu_droplet)
    rate_dir = os.path.join(WORKERS, f"rate_{rate:g}")
    os.makedirs(rate_dir, exist_ok=True)
    print(f"[worker rate={rate:g}] start (pid {os.getpid()})", flush=True)
    t0 = time.time()
    droplet_shear_sweep(
        params, shear_rates=[rate], eq_steps=args.eq_steps,
        shear_steps=args.shear_steps, dt=args.dt, sigma=sigma,
        out_dir=rate_dir,
    )
    print(f"[worker rate={rate:g}] done in {time.time() - t0:.0f}s", flush=True)


def read_worker_row(rate: float) -> dict | None:
    p = os.path.join(WORKERS, f"rate_{rate:g}", "sph_shear_sweep.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    rows = d.get("rows", [])
    for r in rows:
        if abs(float(r["shear_rate_nominal"]) - rate) < 1e-12:
            return r
    return None


def merge(rows: list, traces: dict, args, sigma: float) -> None:
    rows = sorted(rows, key=lambda r: float(r["shear_rate_nominal"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sph_shear_sweep.json"), "w") as f:
        json.dump({
            "params": SPHParams(mu_solvent=args.mu_solvent,
                                mu_droplet=args.mu_droplet).__dict__,
            "domain": [0.0, 0.0, 24.0, 16.0],
            "spacing": 0.5, "droplet_radius": 3.0,
            "rows": rows,
            "note": ("Parallel-sweep merge; each rate is an independent droplet "
                     "simulation. Ca uses the MEASURED local shear rate and the "
                     "Laplace-verified sigma_eff."),
        }, f, indent=2)
    np.savez(os.path.join(OUT, "sph_traces.npz"),
             shear_rates=np.array([float(r["shear_rate_nominal"]) for r in rows]),
             traces=traces)
    # figure + summary (mirrors sph_sweep.py post-processing)
    ca = [r["capillary_number_Ca"] for r in rows]
    ca_nom = [r["capillary_number_nominal"] for r in rows]
    d0 = [r["taylor_initial"] for r in rows]
    dplateau = [r["taylor_plateau_fit"] for r in rows]
    dfinal = [r["taylor_final"] for r in rows]
    zero = [i for i, r in enumerate(rows) if float(r["shear_rate_nominal"]) == 0.0]
    D0 = float(np.mean([d0[i] for i in zero])) if zero else float(np.mean(d0))
    dD_plateau = [p - D0 for p in dplateau]
    dD_final = [f - D0 for f in dfinal]
    dD0 = [i - D0 for i in d0]
    lam = args.mu_droplet / args.mu_solvent
    taylor_coef = (19.0 * lam + 16.0) / (16.0 * lam + 16.0)
    ca_th = np.logspace(np.log10(max(min(ca_nom) * 0.5, 1e-4)),
                        np.log10(max(ca_nom) * 2.0), 40)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ca, dD0, "s--", color="0.6", label="initial (settled) − D$_0$")
    ax.plot(ca, dD_plateau, "o-", color="crimson",
            label="plateau ΔD$_\\infty$ (fit) − D$_0$")
    ax.plot(ca_nom, dD_final, "^:", color="0.4", label="final sample − D$_0$")
    ax.plot(ca_th, taylor_coef * np.asarray(ca_th), "--", color="steelblue",
            label=f"Taylor (1934) small-Ca limit (a={taylor_coef:.2f})")
    ax.set_xlabel("capillary number Ca (measured $\\mu_d\\dot{\\gamma} R/\\sigma$)")
    ax.set_ylabel("excess Taylor deformation ΔD = D − D$_0$")
    ax.set_xscale("log")
    ax.set_title(f"Tau droplet deformation vs shear (2D WCSPH; D$_0$={D0:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sph_deformation_vs_Ca.png"), dpi=150)
    plt.close(fig)
    print(f"\n  [sph] settled baseline D0 = {D0:.4f} (no-shear control)")

    phys = {}
    for tau_pa in (0.1, 1.0):
        ca_phys = tau_pa * 1e-6 / 1e-4
        phys[f"D_at_{tau_pa}Pa"] = ca_phys * taylor_coef
        phys[f"Ca_at_{tau_pa}Pa"] = ca_phys
        print(f"  tau={tau_pa} Pa -> Ca={ca_phys:.4f} -> D ~ {ca_phys * taylor_coef:.4f}")
    with open(os.path.join(OUT, "sph_study_summary.json"), "w") as f:
        json.dump({
            "settled_baseline_D0": D0,
            "equilibration": {"eq_steps": args.eq_steps},
            "sweep_rows": rows,
            "physiology": {
                "note": ("Ca = tau*R/sigma with tau=0.1-1 Pa, R=1e-6 m, sigma=1e-4 "
                         "N/m; physiological D from the analytic Taylor limit."),
                "lambda_viscosity_ratio": lam, **phys,
            },
        }, f, indent=2, default=float)
    print(f"\nDone. Outputs in {OUT}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shear-rates", type=str, default=DEFAULT_RATES)
    ap.add_argument("--eq-steps", type=int, default=4000)
    ap.add_argument("--shear-steps", type=int, default=60000)
    ap.add_argument("--dt", type=float, default=0.008)
    ap.add_argument("--mu-solvent", type=float, default=1.0)
    ap.add_argument("--mu-droplet", type=float, default=10.0)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--rate", type=float, default=None)
    args = ap.parse_args()

    sigma = load_sigma()

    if args.worker:
        if args.rate is None:
            raise SystemExit("--worker requires --rate")
        worker(args.rate, sigma, args)
        return

    rates = [float(x) for x in args.shear_rates.split(",")]
    os.makedirs(WORKERS, exist_ok=True)
    # collect already-completed rows (resume)
    done = {}
    for r in rates:
        row = read_worker_row(r)
        if row is not None:
            done[r] = row
    pending = [r for r in rates if r not in done]
    print(f"resume: {len(done)} done, {len(pending)} pending {pending}")

    procs = []
    i = 0
    while i < len(pending) or procs:
        # (re)fill worker slots
        while len(procs) < args.max_workers and i < len(pending):
            r = pending[i]
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--worker", "--rate", f"{r:g}",
                   "--eq-steps", str(args.eq_steps),
                   "--shear-steps", str(args.shear_steps),
                   "--dt", str(args.dt),
                   "--mu-solvent", str(args.mu_solvent),
                   "--mu-droplet", str(args.mu_droplet)]
            # limit BLAS/OpenMP threads per worker so 6 workers don't thrash
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            procs.append((r, subprocess.Popen(cmd, env=env)))
            i += 1
        # poll
        for r, p in list(procs):
            if p.poll() is not None:
                procs.remove((r, p))
                if p.returncode != 0:
                    print(f"[driver] worker rate={r:g} FAILED rc={p.returncode}")
                else:
                    print(f"[driver] worker rate={r:g} finished")
        if procs:
            time.sleep(2)

    # Build rows entirely from disk: workers write their per-rate JSON only
    # after a rate completes (droplet_shear_sweep checkpoints per rate), so
    # the on-disk files are the single source of truth whether a rate
    # finished in this run, a previous run, or not at all.
    rows = [read_worker_row(r) for r in rates]
    missing = [r for r, row in zip(rates, rows) if row is None]
    if missing:
        raise SystemExit(
            f"sweep incomplete: no result for rates {missing}. "
            "Re-run this script to resume from the per-rate checkpoints.")
    traces = {}
    for r in rates:
        tp = os.path.join(WORKERS, f"rate_{r:g}", "sph_traces.npz")
        if os.path.exists(tp):
            t = np.load(tp, allow_pickle=True)
            tr = t["traces"]
            if hasattr(tr, "item") and getattr(tr, "ndim", 1) == 0:
                tr = tr.item()
            for k, v in tr.items():
                traces[k] = v
    merge(rows, traces, args, sigma)


if __name__ == "__main__":
    main()
