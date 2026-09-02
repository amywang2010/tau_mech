"""Merge per-rate sweep records into the canonical sweep file, with the
pre-registered acceptance checks applied (2026-09-02).

Acceptance criteria (fixed BEFORE the sheared results were observed):

* A0 (control gate): the no-shear control's post-equilibration deformation
  stays bounded: max |D - D0| < 0.02 over the shear-phase trace (mirrors the
  six-criterion zero-shear gate in outputs/sph/audits/
  zero_shear_baseline.json; the G1/G3-G6 criteria are inherited from that
  full-duration record).
* A1 (plateau quality): every sheared case's D(t) fit has R2 >= 0.8 and
  converged = True.
* A2 (monotonicity): D_inf is non-decreasing in Ca_measured.
* A3 (Ca consistency): Ca_measured / Ca_nominal in (0.5, 1.05) - the wall
  slip lowers the measured local shear rate, never raises it above nominal.

Outputs:
  outputs/sph/sph_shear_sweep.json       canonical merged record
  outputs/sph/sph_shear_sweep_summary.json  rows + acceptance verdicts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs" / "sph" / "sweep"


def main() -> None:
    rows, traces = [], {}
    for rd in sorted(SWEEP.glob("rate_*")):
        rec_p = rd / "sph_shear_sweep.json"
        if not rec_p.exists():
            print(f"WARNING: {rd.name} has no record - skipped")
            continue
        rec = json.loads(rec_p.read_text())
        if not rec.get("rows"):
            print(f"WARNING: {rd.name} record empty - skipped")
            continue
        rows.append(rec["rows"][0])
        tr_p = rd / "sph_traces.npz"
        if tr_p.exists():
            t = np.load(tr_p, allow_pickle=True)
            tr = t["traces"].item() if getattr(t["traces"], "ndim", 1) == 0 \
                else t["traces"]
            key = str(rec["rows"][0]["shear_rate_nominal"])
            if key in tr:
                traces[key] = tr[key]
    rows.sort(key=lambda r: r["shear_rate_nominal"])

    # ---- pre-registered acceptance checks --------------------------------
    checks = {}
    ctrl = next((r for r in rows if r["shear_rate_nominal"] == 0.0), None)
    if ctrl is None:
        checks["A0_control_present"] = {"pass": False,
                                        "detail": "no control row"}
    else:
        d = np.asarray(traces.get("0.0", {}).get("taylor", []), dtype=float)
        if len(d) >= 5:
            d0 = float(d[0])
            md = float(np.abs(d - d0).max())
            checks["A0_control_bounded"] = {
                "pass": bool(md < 0.02), "max_abs_D_minus_D0": md,
                "limit": 0.02}
        else:
            checks["A0_control_bounded"] = {
                "pass": False, "detail": "control trace too short"}

    sheared = [r for r in rows if r["shear_rate_nominal"] > 0.0]
    fits = [(r["shear_rate_nominal"], r.get("fit_r2"),
             r.get("fit_converged"), r.get("taylor_plateau_fit"),
             r.get("capillary_number_Ca"),
             r.get("capillary_number_nominal")) for r in sheared]
    checks["A1_plateau_quality"] = {
        "pass": bool(all(f[1] is not None and f[1] >= 0.8 and f[2]
                         for f in fits)),
        "per_case": {f"{f[0]}": {"r2": f[1], "converged": f[2]} for f in fits}}
    d_inf = [(f[4], f[3]) for f in fits if f[3] is not None and f[4] is not None]
    d_inf.sort()
    checks["A2_monotone_D_inf_in_Ca"] = {
        "pass": bool(all(d_inf[i + 1][1] >= d_inf[i][1] - 1e-3
                         for i in range(len(d_inf) - 1))),
        "sequence": [(round(c, 4), round(dd, 4)) for c, dd in d_inf]}
    checks["A3_Ca_consistency"] = {
        "pass": bool(all(0.5 < f[4] / f[5] <= 1.05 for f in fits
                         if f[4] and f[5])),
        "per_case": {f"{f[0]}": (round(f[4] / f[5], 4) if f[4] and f[5] else None)
                     for f in fits}}

    # ---- canonical merged record -----------------------------------------
    meta_p = next(iter(sorted(SWEEP.glob("rate_*/sph_shear_sweep.json"))))
    meta = json.loads(meta_p.read_text())
    merged = {
        "params": meta.get("params", {}),
        "domain": meta.get("domain"), "spacing": meta.get("spacing"),
        "droplet_radius": meta.get("droplet_radius"),
        "dt": meta.get("dt"), "eq_steps": meta.get("eq_steps"),
        "rows": rows,
        "note": ("merged from per-rate runs (wave driver, 2-way parallel); "
                 "validated solver (CSF symmetric stencil, zero-shear gate "
                 "PASS); fresh Laplace calibration sigma_eff = 1.0641; "
                 "defective-solver Aug-15 sweep archived under "
                 "outputs/sph/archive_pre_csffix/"),
    }
    out = ROOT / "outputs" / "sph" / "sph_shear_sweep.json"
    out.write_text(json.dumps(merged, indent=2))
    np.savez(ROOT / "outputs" / "sph" / "sph_traces.npz",
             shear_rates=np.array([float(k) for k in traces]),
             traces=traces)

    summary = {"acceptance_pre_registered": checks,
               "all_acceptance_pass": bool(all(
                   c.get("pass", False) for c in checks.values())),
               "rows": rows}
    (ROOT / "outputs" / "sph" / "sph_shear_sweep_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(checks, indent=1))
    print("ALL_ACCEPTANCE_PASS:", summary["all_acceptance_pass"])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
