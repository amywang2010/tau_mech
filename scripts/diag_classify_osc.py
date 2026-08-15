"""Classify the droplet "shape oscillation" by isolating each force term.

Decisive facts already established (scripts/diag_osc_mode.py -> osc.log):

  * D(t) oscillates with a period/amplitude IDENTICAL across sigma_surf =
    0.5, 1.0, 2.0 (D_min=0.0001, D_max=0.0245, t_min=4.8, t_max=0.8 to four
    decimals). A physical Rayleigh capillary oscillation MUST scale as
    1/sqrt(sigma), so this oscillation is NOT surface-tension-driven physics.
  * The period is also independent of mu_droplet (10 vs 50), i.e. it is not
    damped by the droplet's internal viscosity.

This battery isolates the driver by disabling terms one at a time and by
varying the SOLVENT viscosity (never varied before), while also tracking the
droplet center of mass to separate a genuine shape change from a spurious
translation. Each config runs the R=3 droplet in the standard 24x16 cell and
reports D0, D_min, D_max, t_min, t_max, the COM drift, and a compact trace.

Configs:
  1 baseline       sigma=0 A_surf=0 B_surf=0            (no surface forces)
  2 asurf_only     sigma=0 A_surf=10 B_surf=0
  3 csf_only       sigma=1 A_surf=0  B_surf=0
  4 csf_no_shep    csf_only + shepard=0
  5 csf_no_xsph    csf_only + xsph=0
  6 csf_no_artv    csf_only + alpha_art=0
  7 csf_mus_lo     csf_only + mu_solvent=0.5
  8 csf_mus_hi     csf_only + mu_solvent=2.0
  9 csf_sig_4x     sigma=4 A_surf=0 B_surf=0            (amplitude scaling test)
"""
from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tau_mech.sph import (SPHParams, make_couette_droplet_state, run,
                          droplet_deformation, step)


def trace_config(name: str, params: SPHParams, steps: int = 3000,
                 dt: float = 0.008, every: int = 100):
    state = make_couette_droplet_state(
        params, domain=(0.0, 0.0, 24.0, 16.0), spacing=0.5,
        droplet_radius=3.0, n_wall_layers=4)
    run(state, params, 0, dt)  # density init + mass renormalization only
    Ds, ts = [], []
    com0 = None
    for s in range(steps):
        step(state, params, dt)
        if s % every == 0:
            d = droplet_deformation(state)
            com = np.asarray(d["com"])
            if com0 is None:
                com0 = com.copy()
            Ds.append(d["taylor"])
            ts.append(s * dt)
    Ds = np.array(Ds)
    ts = np.array(ts)
    d_final = droplet_deformation(state)
    com_drift = float(np.linalg.norm(np.asarray(d_final["com"]) - com0))
    imin = int(np.argmin(Ds))
    imax = int(np.argmax(Ds))
    # compact trace of D for the report
    tr = " ".join(f"{v:.3f}" for v in Ds[::max(1, len(Ds)//16)])
    return {"name": name, "D0": Ds[0], "D_min": Ds[imin], "D_max": Ds[imax],
            "t_min": ts[imin], "t_max": ts[imax], "span": Ds.max() - Ds.min(),
            "com_drift": com_drift, "trace": tr}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--configs", type=str, default="all")
    a = ap.parse_args()
    configs = [
        ("baseline", dict(sigma_surf=0.0, A_surf=0.0, B_surf=0.0)),
        ("asurf_only", dict(sigma_surf=0.0, A_surf=10.0, B_surf=0.0)),
        ("csf_only", dict(sigma_surf=1.0, A_surf=0.0, B_surf=0.0)),
        ("csf_no_shep", dict(sigma_surf=1.0, A_surf=0.0, B_surf=0.0, shepard=0.0)),
        ("csf_no_xsph", dict(sigma_surf=1.0, A_surf=0.0, B_surf=0.0, xsph=0.0)),
        ("csf_no_artv", dict(sigma_surf=1.0, A_surf=0.0, B_surf=0.0, alpha_art=0.0)),
        ("csf_mus_lo", dict(sigma_surf=1.0, A_surf=0.0, B_surf=0.0, mu_solvent=0.5)),
        ("csf_mus_hi", dict(sigma_surf=1.0, A_surf=0.0, B_surf=0.0, mu_solvent=2.0)),
        ("csf_sig_4x", dict(sigma_surf=4.0, A_surf=0.0, B_surf=0.0)),
    ]
    if a.configs != "all":
        keep = set(a.configs.split(","))
        configs = [(n, o) for n, o in configs if n in keep]
    print(f"{'config':<14} {'D0':>7} {'D_min':>7} {'D_max':>7} "
          f"{'t_min':>6} {'t_max':>6} {'span':>7} {'com_drift':>9}")
    rows = []
    for name, overrides in configs:
        params = SPHParams(mu_solvent=1.0, mu_droplet=10.0)
        for k, v in overrides.items():
            setattr(params, k, v)
        r = trace_config(name, params, steps=a.steps)
        rows.append(r)
        print(f"{name:<14} {r['D0']:7.4f} {r['D_min']:7.4f} {r['D_max']:7.4f} "
              f"{r['t_min']:6.1f} {r['t_max']:6.1f} {r['span']:7.4f} "
              f"{r['com_drift']:9.4f}")
    print("\n--- D(t) traces (16 samples over 24 units) ---")
    for r in rows:
        print(f"{r['name']:<14} {r['trace']}")


if __name__ == "__main__":
    main()
