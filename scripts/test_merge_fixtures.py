"""Sandbox verification harness for merge_final_sweep.py v1.4.

Builds throwaway sandboxes from the REAL wave-1/2 records (never touches
canonical outputs) and exercises every v1.4 branch:

  S1  v1.4 classification on real data (window_limited expected for 0.03:
      tau=116 > T/2=112).
  S2  extension supersession with a consistent extension trace (A5 pass,
      trusted fit, verdict upgraded).
  S3  extension with an inconsistent trace (A5 fail -> overall FAIL).

Run:  python scripts/test_merge_fixtures.py
Exit code 0 iff every scenario behaves as specified.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
REAL_SWEEP = ROOT / "outputs" / "sph" / "sweep"
MERGE = ROOT / "scripts" / "merge_final_sweep.py"


def load_traces(rd: Path) -> dict:
    t = np.load(rd / "sph_traces.npz", allow_pickle=True)
    tr = t["traces"]
    return tr.item() if getattr(tr, "ndim", 1) == 0 else tr


def make_ext_dir(dst: Path, src: Path, delta: float, n_ext: int = 75) -> None:
    """Copy a rate dir into an ext_ dir; extend its trace by n_ext samples.

    delta: offset applied to the ENTIRE extension trace. This models
    trajectory divergence between two independent runs of the same rate
    (nondeterminism / different initial conditions), which manifests on the
    shared span where A5 has power. Offsetting only the (unobservable)
    tail would be undetectable by ANY shared-span check by construction.
    delta = 0 -> same deterministic trajectory (A5 must pass); delta > 2N
    -> integrity violation (A5 must fail).
    """
    shutil.copytree(src, dst)
    rec = json.loads((dst / "sph_shear_sweep.json").read_text())
    tr = load_traces(dst)
    key = str(rec["rows"][0]["shear_rate_nominal"])
    base = tr[key]
    t = list(base["t"])
    d = [x + delta for x in base["taylor"]]
    dt_t = (t[-1] - t[0]) / (len(t) - 1)
    tail_t = [t[-1] + k * dt_t for k in range(1, n_ext + 1)]
    # consistent continuation: flat at the (offset) last sample
    tail_d = [d[-1]] * n_ext
    tr[key] = {"t": t + tail_t,
               "taylor": d + tail_d,
               "aspect_ratio": list(base["aspect_ratio"]) + [base["aspect_ratio"][-1]] * n_ext,
               "angle_deg": list(base["angle_deg"]) + [base["angle_deg"][-1]] * n_ext,
               "gamma_dot_measured": list(base["gamma_dot_measured"]) + [base["gamma_dot_measured"][-1]] * n_ext,
               "Ca_measured": list(base["Ca_measured"]) + [base["Ca_measured"][-1]] * n_ext}
    np.savez(dst / "sph_traces.npz",
             shear_rates=np.array([float(key)]), traces=tr)
    # give the extension row a short fitted tau so the trust rule can pass
    rec["rows"][0]["tau_transient"] = 40.0
    rec["rows"][0]["fit_converged"] = True
    rec["rows"][0]["taylor_plateau_fit"] = d[-1]
    rec["rows"][0]["taylor_final"] = d[-1]
    (dst / "sph_shear_sweep.json").write_text(json.dumps(rec, indent=2))


def run_merge(sandbox: Path) -> tuple[dict, dict]:
    """Returns (summary, merged_record)."""
    out = sandbox / "out"
    r = subprocess.run([str(PY), "-u", str(MERGE),
                        "--sweep-dir", str(sandbox),
                        "--out-dir", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"merge failed rc={r.returncode}")
    summ = json.loads((out / "sph_shear_sweep_summary.json").read_text())
    merged = json.loads((out / "sph_shear_sweep.json").read_text())
    return summ, merged


def expect(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="merge_v14_"))
    print(f"sandbox root: {tmp}")

    # ---- S1: real data, v1.4 classification --------------------------------
    print("S1: v1.4 classification on real wave-1/2 records")
    s1 = tmp / "s1"
    shutil.copytree(REAL_SWEEP, s1, ignore=shutil.ignore_patterns("ext_*"))
    summ, merged = run_merge(s1)
    cls = summ["acceptance_notes"]["classification"]
    expect(cls["0.03"]["class"] == "window_limited",
           "rate 0.03 classified window_limited (tau=116 > T/2=112)")
    expect("D_inf_interval" in cls["0.03"],
           "rate 0.03 carries D_inf interval [final, fit]")
    expect(summ["verdict_class"] == "PASS_WITH_LIMITS",
           f"verdict_class = PASS_WITH_LIMITS (got {summ['verdict_class']})")
    expect(summ["acceptance_pre_registered"]
           ["A1d_window_limited_reported_as_interval"]["pass"],
           "A1d passes on real 0.03 data")

    # ---- S2: consistent extension supersedes, A5 passes --------------------
    print("S2: consistent extension (delta=0) supersedes the short window")
    s2 = tmp / "s2"
    shutil.copytree(s1, s2)
    make_ext_dir(s2 / "ext_0.03", s1 / "rate_0.03", delta=0.0)
    summ, merged = run_merge(s2)
    expect("0.03" in merged.get("extended_rates", []),
           "0.03 recorded as superseded by extension")
    expect(summ["acceptance_pre_registered"]
           ["A5_extension_prefix_consistency"]["pass"],
           "A5 passes: extension trace matches short window within 2N")
    row003 = [r for r in summ["rows"]
              if r["shear_rate_nominal"] == 0.03][0]
    expect(row003["n_shear_steps"] * 0.008 >= 2 * row003["tau_transient"],
           "superseded row passes the v1.4 trust rule")

    # ---- S3: inconsistent extension trips A5 -> FAIL -----------------------
    print("S3: corrupted extension (delta=5N) must FAIL the verdict")
    s3 = tmp / "s3"
    shutil.copytree(s1, s3)
    noise = summ["acceptance_notes"]["control_reference"]["noise_floor_N"]
    make_ext_dir(s3 / "ext_0.03", s1 / "rate_0.03", delta=5.0 * noise)
    summ3, _ = run_merge(s3)
    expect(summ3["acceptance_pre_registered"]
           ["A5_extension_prefix_consistency"]["pass"] is False,
           "A5 correctly fails on a 5N-discrepant extension trace")
    expect(summ3["all_acceptance_pass"] is False
           and summ3["verdict_class"] == "FAIL",
           "overall verdict FAIL (gating unchanged)")

    # ---- S4: genuine non-monotonicity must still FAIL A2 --------------------
    print("S4: interval-feasibility A2 still detects genuine violations")
    s4 = tmp / "s4"
    shutil.copytree(s1, s4)
    # Make 0.03 look trusted WITHOUT changing its measured plateau (0.743):
    # the measured sequence 0.743 (Ca~0.5) -> 0.689 (Ca~1.2) is genuinely
    # non-monotone, and A2 must fail on it.
    rec4 = json.loads((s4 / "rate_0.03" / "sph_shear_sweep.json").read_text())
    rec4["rows"][0]["tau_transient"] = 40.0
    rec4["rows"][0]["fit_converged"] = True
    (s4 / "rate_0.03" / "sph_shear_sweep.json").write_text(
        json.dumps(rec4, indent=2))
    summ4, _ = run_merge(s4)
    expect(summ4["acceptance_pre_registered"]
           ["A2_monotone_D_inf_in_Ca_distinguishable"]["pass"] is False,
           "A2 fails on a genuinely non-monotone trusted sequence")
    expect(summ4["verdict_class"] == "FAIL", "verdict FAIL (S4)")

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL SCENARIOS PASS")


if __name__ == "__main__":
    main()
