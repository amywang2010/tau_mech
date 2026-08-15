"""Sample per-process CPU deltas (user time) over ~15 s.

Helps find which processes actually consume cores on this machine.
"""
import subprocess
import time


def sample():
    out = subprocess.run(
        ["wmic", "process", "get", "ProcessId,Name,UserModeTime", "/format:csv"],
        capture_output=True, text=True, errors="ignore").stdout
    d = {}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[3].strip().isdigit():
            d[int(parts[3].strip())] = (parts[1].strip(), int(parts[2].strip() or 0))
    return d


s1 = sample()
time.sleep(15)
s2 = sample()
print(f"{'name':<14}{'pid':>8}{'delta_s':>10}{'cores':>8}")
for pid, (name, t2) in s2.items():
    if pid not in s1:
        continue
    t1 = s1[pid][1]
    dt = (t2 - t1) / 1e7  # 100 ns units -> seconds
    if dt > 0.5:
        print(f"{name:<14}{pid:>8}{dt:>10.1f}{dt / 15:>8.2f}")
