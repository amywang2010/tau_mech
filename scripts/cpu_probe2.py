"""Sample CPU accumulation of the diag process over 15 s (progress check)."""
import subprocess
import time

PID = 28488


def cpu() -> int:
    out = subprocess.run(
        ["wmic", "process", "where", f"processid={PID}", "get", "UserModeTime",
         "/format:list"],
        capture_output=True, text=True, errors="ignore").stdout
    for line in out.splitlines():
        if line.startswith("UserModeTime="):
            return int(line.split("=")[1] or 0)
    return -1


c1 = cpu()
time.sleep(15)
c2 = cpu()
dt = (c2 - c1) / 1e7  # 100 ns units -> seconds
print(f"CPU delta over 15 s = {dt:.2f} s  ->  {dt / 15:.2f} cores in use")
