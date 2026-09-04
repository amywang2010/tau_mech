"""Autonomous post-completion chain for resolution study v2 (2026-09-03).

Executes ONLY pre-declared mechanical steps after the study process exits;
no scientific interpretation happens here:

  1. Wait for couette_resolution_study.json to be rewritten (mtime newer
     than the watcher start) AND no python process running
     diag_couette_resolution.py. Polls every 2 min; 16 h safety timeout.
  2. Load the record.
     - attribution present  -> rebuild the strict final report, run the
       test suite, commit record + report + push.
     - attribution is None (aborted) -> commit the abort record + push
       (data preservation first); report rebuild is left for review.
  3. Any failure is logged and never deletes data; the repository always
     ends up containing the raw record.

Run:  python scripts/watch_resolution_v2.py   (background, nohup)
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "outputs/sph/audits/couette_resolution_study.json"
REPORT = ROOT / "outputs/final_report.md"
LOG = ROOT / "outputs/sph/logs/watcher_v2.log"
POLL_S = 120
TIMEOUT_S = 16 * 3600


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          **kw)


def study_running() -> bool:
    r = sh(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object -ExpandProperty CommandLine"])
    return "diag_couette_resolution" in (r.stdout or "")


def main() -> None:
    log("watcher armed; waiting for resolution study v2.1 (--level1) to finish")
    t0 = time.time()
    marker = REC.stat().st_mtime if REC.exists() else 0.0
    restarts = 0
    while True:
        if REC.exists() and REC.stat().st_mtime > marker \
                and not study_running():
            log("record rewritten and study process gone -> complete")
            break
        if not study_running() and REC.stat().st_mtime <= marker:
            # Study died WITHOUT writing a record (crash / OOM / power).
            # One documented restart; if it dies again, leave everything
            # for morning review (repeated blind restarts are not
            # scientific behavior).
            if restarts < 1:
                restarts += 1
                log("study process gone WITHOUT a new record - "
                    "documented restart attempt 1/1")
                r = subprocess.Popen(
                    [str(ROOT / ".venv/Scripts/python.exe"), "-u",
                     "scripts/diag_couette_resolution.py", "--level1"],
                    cwd=ROOT,
                    stdout=open(ROOT / "outputs/sph/logs/"
                                "resolution_v21_retry.log", "w"),
                    stderr=subprocess.STDOUT)
                log(f"restarted as PID {r.pid}")
            else:
                log("study gone again without a record - NOT restarting "
                    "twice; leaving state for manual review")
                return
        if time.time() - t0 > TIMEOUT_S:
            log("TIMEOUT after 16 h - exiting without action; record "
                "will be handled manually")
            return
        time.sleep(POLL_S)

    rec = json.loads(REC.read_text())
    attr = rec.get("attribution")

    if attr is None:
        log(f"study ABORTED: {rec.get('aborted', 'unstated')!r}; "
            "committing the abort record (data preservation)")
        sh(["git", "add", str(REC.relative_to(ROOT))])
        c = sh(["git", "commit", "-m",
                "Resolution study v2 aborted: "
                f"{rec.get('aborted', 'unstated')} "
                "(record committed for the evidence chain)\n\n"
                "\U0001F916 Generated with Codebuff\n"
                "Co-Authored-By: Codebuff <noreply@codebuff.com>"])
        log(f"commit rc={c.returncode}")
        p = sh(["git", "push"])
        log(f"push rc={p.returncode}")
        return

    log(f"attribution present: monotone="
        f"{attr.get('monotone_decreasing_with_refinement')}; rebuilding "
        f"strict report")
    b = sh([str(ROOT / ".venv/Scripts/python.exe"),
            "scripts/build_final_report.py"])
    log(f"report build rc={b.returncode}"
        + ("" if b.returncode == 0 else f" stderr={b.stderr[-400:]!r}"))
    if b.returncode == 0:
        t = sh([str(ROOT / ".venv/Scripts/python.exe"), "-m", "pytest",
                "tests/", "-q"])
        log(f"pytest rc={t.returncode}")
        sh(["git", "add", str(REC.relative_to(ROOT)),
            str(REPORT.relative_to(ROOT))])
        mono = attr.get("monotone_decreasing_with_refinement")
        c = sh(["git", "commit", "-m",
                f"Resolution study v2 complete: monotone_decrease={mono}; "
                "report regenerated (strict)\n\n"
                "\U0001F916 Generated with Codebuff\n"
                "Co-Authored-By: Codebuff <noreply@codebuff.com>"])
        log(f"commit rc={c.returncode}")
        p = sh(["git", "push"])
        log(f"push rc={p.returncode}")
    else:
        # Report build failed - still preserve the raw record in git.
        log("report build FAILED; committing the raw record only "
            "(report review left for the morning)")
        sh(["git", "add", str(REC.relative_to(ROOT))])
        sh(["git", "commit", "-m",
            "Resolution study v2 record (report build pending review)\n\n"
            "\U0001F916 Generated with Codebuff\n"
            "Co-Authored-By: Codebuff <noreply@codebuff.com>"])
        sh(["git", "push"])
    log("watcher done")


if __name__ == "__main__":
    main()
