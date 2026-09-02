"""Autonomous driver for the final physiological shear sweep (2026-09-02).

Design (pre-registered before any sheared result was observed; v1.1):

* Rates: 0.0 (no-shear CONTROL) + 0.001, 0.003, 0.01, 0.03, 0.1.
* Protocol per rate: eq 4000 steps + 28000 shear steps (t = 224 units
  = 8 capillary times t_char ~ 28; the exponential plateau residual at 8
  t_char is e^-8 ~ 3e-4, so the D_inf fit is fully constrained; the
  no-shear control uses the SAME window so control vs sheared cases are
  directly comparable, and the full-duration 50765-step zero-shear gate
  record already covers the longer window for the control).
* Waves of 2 concurrent rates (evidence-based contention level: 2-way
  parallelism measured 1.45x throughput vs serial on this machine).
  Wave 1 = control + extreme rate; later waves only proceed if the wave-1
  control record exists (the gate criterion itself is applied at merge).
* After the sweep: the Couette wall-slip resolution study (production,
  3 resolutions), then the merge + pre-registered acceptance checks
  (scripts/merge_final_sweep.py), then the full test suite.
* Every stage is checkpointed: completed rates live in their own out_dir
  and are never recomputed on a driver restart (the runner resumes from
  the per-rate record).

The driver is launched detached (nohup) and survives terminal/IDE close;
AC standby is disabled on this machine (powercfg /a verified).

Usage:
    python scripts/run_final_sweep_driver.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SWEEP = ROOT / "outputs" / "sph" / "sweep"
LOGS = ROOT / "logs"

WAVES = [["0.0", "0.1"], ["0.001", "0.03"], ["0.003", "0.01"]]
EQ, SHEAR = 4000, 28000


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def rate_dir(rate: str) -> Path:
    return SWEEP / f"rate_{rate}"


def run_wave(rates: list[str]) -> dict:
    """Launch one wave (2 rates in parallel) and wait for both."""
    procs = {}
    for rate in rates:
        rd = rate_dir(rate)
        rd.mkdir(parents=True, exist_ok=True)
        calib = ROOT / "outputs" / "sph" / "laplace_calibration.json"
        if not (rd / "laplace_calibration.json").exists():
            shutil.copy(calib, rd / "laplace_calibration.json")
        logf = open(LOGS / f"sweep_r{rate}.log", "w")
        p = subprocess.Popen(
            [str(PY), "-u", str(ROOT / "scripts" / "run_one_rate.py"),
             rate, str(rd), "--eq", str(EQ), "--shear", str(SHEAR)],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        procs[rate] = (p, logf)
        log(f"launched rate {rate} (pid {p.pid})")
    status = {}
    for rate, (p, logf) in procs.items():
        rc = p.wait()
        logf.close()
        status[rate] = {"returncode": rc,
                        "done": rc == 0}
        log(f"rate {rate} finished rc={rc}")
    return status


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    record = {"start": time.strftime("%Y-%m-%d %H:%M:%S"),
              "eq_steps": EQ, "shear_steps": SHEAR, "waves": WAVES,
              "rates": {}}
    for wave in WAVES:
        log(f"=== wave {wave} ===")
        record["rates"].update(run_wave(wave))
        (SWEEP / "driver_record.json").write_text(json.dumps(record, indent=2))
        if "0.0" in wave and not record["rates"]["0.0"]["done"]:
            log("CONTROL FAILED - aborting remaining waves")
            break

    # Resolution study (production, single job).
    log("=== Couette resolution study (production) ===")
    with open(LOGS / "couette_resolution.log", "w") as logf:
        rc = subprocess.Popen(
            [str(PY), "-u", str(ROOT / "scripts" / "diag_couette_resolution.py")],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT)).wait()
    record["resolution_rc"] = rc
    log(f"resolution study rc={rc}")

    # Merge + pre-registered acceptance checks.
    log("=== merge + acceptance ===")
    with open(LOGS / "merge.log", "w") as logf:
        rc = subprocess.Popen(
            [str(PY), "-u", str(ROOT / "scripts" / "merge_final_sweep.py")],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT)).wait()
    record["merge_rc"] = rc
    log(f"merge rc={rc}")

    # Full test suite.
    log("=== test suite ===")
    with open(LOGS / "driver_tests.log", "w") as logf:
        rc = subprocess.Popen(
            [str(PY), "-m", "pytest", "tests/", "-q"],
            stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT)).wait()
    record["tests_rc"] = rc
    record["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (SWEEP / "driver_record.json").write_text(json.dumps(record, indent=2))
    log(f"DRIVER COMPLETE rc={rc}")


if __name__ == "__main__":
    main()
