"""Autonomous extension orchestrator (v1.4 protocol, pre-registered 2026-09-03).

Runs AFTER the wave driver finishes (wave-3 rates 0.003/0.01 must exist).
Per the v1.4 amendment (commit 4500298, before wave-3 data existed):

  1. Wait for rate_0.003 and rate_0.01 records (poll; also works if the
     driver already finished).
  2. Apply the v1.4 trust rule to every sheared rate: trusted iff
     fit_converged AND T_window >= 2*tau_fit. Window-limited cases with
     T_ext = 3*tau_fit <= 1200 time units (committed feasibility cap,
     150000 steps) are re-run from scratch in ext_<rate>/ with the
     extended window. All extensions run concurrently (contention is
     preferable to wall-clock time on a machine that may sleep).
  3. Re-run the merge with pre-registered acceptance checks, then the
     full test suite.
  4. Write outputs/sph/sweep/extension_record.json documenting every
     decision (tau, T_ext, feasible/launched/skipped + reason).

Detached (nohup) and checkpointed; survives terminal/IDE close.

Usage:  python scripts/run_extension_orchestrator.py
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SWEEP = ROOT / "outputs" / "sph" / "sweep"
LOGS = ROOT / "logs"
DT = 0.008
CAP_UNITS = 1200.0        # committed feasibility cap (v1.4), time units
CAP_STEPS = 150000        # = CAP_UNITS / DT
WAIT_TARGETS = {"0.003", "0.01"}
MAX_WAIT_S = 6 * 3600


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_row(rd: Path) -> dict | None:
    p = rd / "sph_shear_sweep.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    return rec["rows"][0] if rec.get("rows") else None


def trusted(row: dict) -> bool:
    tau = row.get("tau_transient")
    T = row.get("n_shear_steps", 0) * DT
    return bool(row.get("fit_converged")) and tau is not None \
        and math.isfinite(tau) and tau > 0 and T >= 2.0 * tau


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    record = {"start": time.strftime("%Y-%m-%d %H:%M:%S"),
              "protocol": "v1.4 (commit 4500298, pre-registered before wave-3)",
              "cap_units": CAP_UNITS, "decisions": {}}

    # ---- 1. wait for wave-3 -------------------------------------------------
    deadline = time.time() + MAX_WAIT_S
    while time.time() < deadline:
        missing = [r for r in WAIT_TARGETS if load_row(SWEEP / f"rate_{r}") is None]
        if not missing:
            break
        log(f"waiting for wave-3 records; missing {missing}")
        time.sleep(300)
    else:
        record["error"] = "wave-3 records did not appear within MAX_WAIT_S"
        (SWEEP / "extension_record.json").write_text(json.dumps(record, indent=2))
        log(record["error"])
        sys.exit(1)

    # ---- 2. trust classification + extension decisions ---------------------
    to_run = []
    for rd in sorted(SWEEP.glob("rate_*")):
        rate = rd.name.replace("rate_", "")
        row = load_row(rd)
        if row is None or row.get("shear_rate_nominal", 0) <= 0:
            continue
        tau = row.get("tau_transient")
        T = row.get("n_shear_steps", 0) * DT
        if trusted(row):
            record["decisions"][rate] = {
                "verdict": "trusted (no extension needed)",
                "tau": tau, "T_window": T}
            continue
        if tau is None or not math.isfinite(tau) or tau <= 0:
            record["decisions"][rate] = {
                "verdict": "no usable tau (not converged) - remains "
                           "window_limited/below-floor per merge rules",
                "tau": tau, "T_window": T}
            continue
        T_ext = 3.0 * tau
        if T_ext > CAP_UNITS:
            record["decisions"][rate] = {
                "verdict": f"extension infeasible: 3*tau={T_ext:.0f} > cap "
                           f"{CAP_UNITS:.0f} - remains window_limited with "
                           "bounded interval",
                "tau": tau, "T_ext_units": T_ext}
            continue
        steps = min(int(math.ceil(T_ext / DT)), CAP_STEPS)
        record["decisions"][rate] = {
            "verdict": "extension launched", "tau": tau,
            "T_ext_units": T_ext, "steps": steps,
            "out_dir": str(SWEEP / f"ext_{rate}")}
        to_run.append((rate, steps))

    # ---- 3. run extensions (all concurrent) --------------------------------
    procs = {}
    for rate, steps in to_run:
        rd = SWEEP / f"ext_{rate}"
        rd.mkdir(parents=True, exist_ok=True)
        calib = ROOT / "outputs" / "sph" / "laplace_calibration.json"
        if not (rd / "laplace_calibration.json").exists():
            shutil.copy(calib, rd / "laplace_calibration.json")
        logf = open(LOGS / f"ext_r{rate}.log", "w")
        p = subprocess.Popen(
            [str(PY), "-u", str(ROOT / "scripts" / "run_one_rate.py"),
             rate, str(rd), "--eq", "4000", "--shear", str(steps)],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        procs[rate] = (p, logf)
        log(f"launched extension rate {rate} steps={steps} pid={p.pid}")
    for rate, (p, logf) in procs.items():
        rc = p.wait()
        logf.close()
        record["decisions"][rate]["returncode"] = rc
        log(f"extension {rate} rc={rc}")

    # ---- 4. merge + tests ---------------------------------------------------
    with open(LOGS / "merge.log", "w") as logf:
        rc = subprocess.Popen(
            [str(PY), "-u", str(ROOT / "scripts" / "merge_final_sweep.py")],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT)).wait()
    record["merge_rc"] = rc
    log(f"merge rc={rc}")
    with open(LOGS / "driver_tests.log", "a") as logf:
        rc = subprocess.Popen(
            [str(PY), "-m", "pytest", "tests/", "-q"],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT)).wait()
    record["tests_rc"] = rc
    record["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (SWEEP / "extension_record.json").write_text(json.dumps(record, indent=2))
    log(f"EXTENSION ORCHESTRATOR COMPLETE tests_rc={rc}")


if __name__ == "__main__":
    main()
