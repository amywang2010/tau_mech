"""Merge per-rate sweep records into the canonical sweep file, with the
pre-registered acceptance checks applied.

v1.2 amendment (2026-09-02 21:40, committed BEFORE waves 2-3 completed):

  Basis: the rate-0.1 fit (a = D_inf/Ca = 0.3815, measured on completed
  wave 1) predicts, at the smallest physiological rate 0.001
  (Ca_nom = 0.0282), a sustained deformation D_inf ~ 0.007 -- SMALLER than
  the droplet's own unforced residual transient (the control trace relaxes
  from D0 = 0.0098 to 0.0004 over the identical window). An exponential
  transient fit to a trace already at steady state is statistically
  ill-posed (cf. the control row: a = -9.4e9, tau = nan, R2 = 0). Applying
  the original A1 (R2 >= 0.8 AND converged) verbatim to such a case would
  fail it for reasons unrelated to physics or solver validity.

  Rule (metrological censoring; the standard "below limit of detection"
  convention). All quantities are derived from data -- no free parameters:

    * D_ctrl_sustained := mean of the last 20% of the control trace
      (fixed fractional rule; 25 samples -> last 5).
    * Noise floor N := max_t |D_ctrl(t) - D_ctrl_sustained|  -- the largest
      excursion the droplet exhibits with ZERO applied shear over the
      identical window, measured from the control trace alone.
    * Signal S := |D_inf,case - D_ctrl_sustained|.
    * If S < N: the case is classified `below_noise_floor`. It is reported
      with a censored upper bound D_inf < D_ctrl_sustained + N (never as a
      failed measurement), and A1 is replaced by
        A1b: the case's ENTIRE trace must lie inside the control envelope,
             max_t |D_case(t) - D_ctrl_sustained| <= N (a machine-epsilon
             guard is added: both runs start from the same deterministic
             equilibrated state, so equality is expected to machine
             precision, not exceeded).
    * A1 (R2 >= 0.8 AND fit converged) applies only to cases classified
      `signal_distinguishable`.
    * A2 monotonicity is tested over the distinguishable cases (the ordering
      of censored values carries no information); consistency of the
      censored set is additionally required: every censored plateau
      estimate must be < min(distinguishable D_inf) + N.

Original pre-registered checks (v1.1, unchanged):

  A0 (control gate): max |D - D0| < 0.02 over the control's shear-phase
      trace (mirrors the six-criterion zero-shear gate in
      outputs/sph/audits/zero_shear_baseline.json).
  A3 (Ca consistency): Ca_measured / Ca_nominal in (0.5, 1.05) -- the wall
      slip lowers the measured local shear rate, never raises it above
      nominal.

Outputs (defaults; override only for sandbox dry-runs):
  outputs/sph/sph_shear_sweep.json           canonical merged record
  outputs/sph/sph_shear_sweep_summary.json   rows + acceptance verdicts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS_GUARD = 1e-12  # machine-epsilon guard on A1b equality, not a tuned parameter


def classify_and_check(rows: list[dict], traces: dict) -> tuple[dict, dict]:
    """Apply the v1.2 pre-registered acceptance checks. Returns (checks, notes)."""
    checks: dict = {}
    notes: dict = {}

    # ---- control-derived reference quantities (no free parameters) --------
    ctrl_trace = traces.get("0.0", {})
    d_c = np.asarray(ctrl_trace.get("taylor", []), dtype=float)
    if len(d_c) < 5:
        return {"A0_control_present": {"pass": False,
                                       "detail": "control trace missing/short"}}, notes
    n_tail = max(1, len(d_c) // 5)  # last 20%
    d_ctrl_sust = float(np.mean(d_c[-n_tail:]))
    noise_floor = float(np.abs(d_c - d_ctrl_sust).max())
    notes["control_reference"] = {
        "D_ctrl_sustained": d_ctrl_sust,
        "noise_floor_N": noise_floor,
        "rule": "N = max|D_ctrl(t) - mean(last 20% of control trace)|",
    }

    # ---- A0: control boundedness (v1.1, unchanged) ------------------------
    d0 = float(d_c[0])
    md = float(np.abs(d_c - d0).max())
    checks["A0_control_bounded"] = {"pass": bool(md < 0.02),
                                    "max_abs_D_minus_D0": md, "limit": 0.02}

    # ---- per-case classification -----------------------------------------
    sheared = [r for r in rows if r["shear_rate_nominal"] > 0.0]
    classification = {}
    for r in sheared:
        rate = r["shear_rate_nominal"]
        key = str(rate)
        d = np.asarray(traces.get(key, {}).get("taylor", []), dtype=float)
        plateau = r.get("taylor_plateau_fit")
        if plateau is None or len(d) < 5:
            classification[key] = {"class": "unclassifiable",
                                   "detail": "missing plateau fit or short trace"}
            continue
        signal = abs(float(plateau) - d_ctrl_sust)
        cls = ("signal_distinguishable" if signal >= noise_floor
               else "below_noise_floor")
        entry = {"class": cls, "signal_S": signal,
                 "signal_to_noise": signal / noise_floor,
                 "D_inf_censored_upper": (d_ctrl_sust + noise_floor
                                          if cls == "below_noise_floor" else None)}
        if cls == "below_noise_floor":
            env = float(np.abs(d - d_ctrl_sust).max())
            entry["A1b_max_abs_dev"] = env
            entry["A1b_pass"] = bool(env <= noise_floor * (1.0 + EPS_GUARD)
                                     + 1e-15)
        classification[key] = entry
    notes["classification"] = classification

    # ---- A1 (v1.2): plateau quality for distinguishable cases; A1b envelope
    a1_cases, a1b_cases = {}, {}
    for r in sheared:
        key = str(r["shear_rate_nominal"])
        c = classification.get(key, {})
        if c.get("class") == "signal_distinguishable":
            a1_cases[key] = {"r2": r.get("fit_r2"),
                             "converged": r.get("fit_converged")}
        elif c.get("class") == "below_noise_floor":
            a1b_cases[key] = c.get("A1b_pass", False)
    checks["A1_plateau_quality_distinguishable"] = {
        "pass": bool(a1_cases) and all(
            v["r2"] is not None and v["r2"] >= 0.8 and v["converged"]
            for v in a1_cases.values()),
        "per_case": a1_cases,
        "n_distinguishable": len(a1_cases),
        "n_below_noise_floor": len(a1b_cases)}
    if a1b_cases:
        checks["A1b_below_floor_within_control_envelope"] = {
            "pass": bool(all(a1b_cases.values())), "per_case": a1b_cases}
    # No case may escape classification (pass-by-omission guard).
    unclassified = [k for k, c in classification.items()
                    if c.get("class") not in ("signal_distinguishable",
                                              "below_noise_floor")]
    checks["A1c_all_sheared_cases_classified"] = {
        "pass": bool(len(unclassified) == 0), "unclassified": unclassified}

    # ---- A2 (v1.2): monotonicity over distinguishable cases ---------------
    d_inf = [(r["capillary_number_Ca"], r["taylor_plateau_fit"])
             for r in sheared
             if classification.get(str(r["shear_rate_nominal"]), {}).get("class")
             == "signal_distinguishable"
             and r.get("capillary_number_Ca") is not None
             and r.get("taylor_plateau_fit") is not None]
    d_inf.sort()
    checks["A2_monotone_D_inf_in_Ca_distinguishable"] = {
        "pass": bool(all(d_inf[i + 1][1] >= d_inf[i][1] - 1e-3
                         for i in range(len(d_inf) - 1))),
        "sequence": [(round(c, 4), round(dd, 4)) for c, dd in d_inf]}
    censored = [(r["capillary_number_Ca"], r["taylor_plateau_fit"])
                for r in sheared
                if classification.get(str(r["shear_rate_nominal"]), {}).get("class")
                == "below_noise_floor"
                and r.get("capillary_number_Ca") is not None
                and r.get("taylor_plateau_fit") is not None]
    if censored and d_inf:
        min_dist = min(dd for _, dd in d_inf)
        checks["A2c_censored_set_consistency"] = {
            "pass": bool(all(dd < min_dist + noise_floor for _, dd in censored)),
            "censored": [(round(c, 4), round(dd, 4)) for c, dd in censored],
            "min_distinguishable_D_inf": min_dist}

    # ---- A3 (v1.1, unchanged): Ca consistency -----------------------------
    fits = [(r["shear_rate_nominal"], r.get("capillary_number_Ca"),
             r.get("capillary_number_nominal")) for r in sheared]
    checks["A3_Ca_consistency"] = {
        "pass": bool(all(f[1] and f[2] and 0.5 < f[1] / f[2] <= 1.05
                         for f in fits)),
        "per_case": {f"{f[0]}": (round(f[1] / f[2], 4) if f[1] and f[2] else None)
                     for f in fits}}
    return checks, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=str(ROOT / "outputs" / "sph" / "sweep"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "sph"))
    args = ap.parse_args()
    sweep_dir, out_dir = Path(args.sweep_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, traces = [], {}
    for rd in sorted(sweep_dir.glob("rate_*")):
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

    checks, notes = classify_and_check(rows, traces)

    # ---- canonical merged record -----------------------------------------
    meta_p = next(iter(sorted(sweep_dir.glob("rate_*/sph_shear_sweep.json"))))
    meta = json.loads(meta_p.read_text())
    merged = {
        "params": meta.get("params", {}),
        "domain": meta.get("domain"), "spacing": meta.get("spacing"),
        "droplet_radius": meta.get("droplet_radius"),
        "dt": meta.get("dt"), "eq_steps": meta.get("eq_steps"),
        "rows": rows,
        "acceptance_version": "v1.2 (below-noise-floor censoring amended "
                              "pre-registered 2026-09-02 21:40, committed "
                              "before wave-2 completion)",
        "note": ("merged from per-rate runs (wave driver, 2-way parallel); "
                 "validated solver (CSF symmetric stencil, zero-shear gate "
                 "PASS); fresh Laplace calibration sigma_eff = 1.0641; "
                 "defective-solver Aug-15 sweep archived under "
                 "outputs/sph/archive_pre_csffix/"),
    }
    out = out_dir / "sph_shear_sweep.json"
    out.write_text(json.dumps(merged, indent=2))
    np.savez(out_dir / "sph_traces.npz",
             shear_rates=np.array([float(k) for k in traces]),
             traces=traces)

    summary = {"acceptance_pre_registered": checks,
               "acceptance_notes": notes,
               "all_acceptance_pass": bool(all(
                   c.get("pass", False) for c in checks.values())),
               "rows": rows}
    (out_dir / "sph_shear_sweep_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps({"checks": checks, "notes": notes}, indent=1, default=str))
    print("ALL_ACCEPTANCE_PASS:", summary["all_acceptance_pass"])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
