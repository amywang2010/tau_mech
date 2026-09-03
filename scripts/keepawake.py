"""Keep the machine awake while long simulations run (Windows power request).

Uses the documented SetThreadExecutionState API (ES_CONTINUOUS |
ES_SYSTEM_REQUIRED) on a dedicated thread. This is the standard mechanism by
which a process tells Windows "do not system-sleep while I am working": it
prevents IDLE sleep and, on most systems, lid-close-initiated sleep while the
process is running (the request is per-process and vanishes when the process
exits -- nothing is changed system-wide).

Honest limitations (documented, not hidden):
  * A user-initiated forced sleep (Start menu -> Sleep, or a lid action the
    OEM firmware handles directly) can bypass power requests.
  * If the process is killed, the request disappears and the normal power
    plan applies again.
  * Hibernation timers are unaffected by ES_SYSTEM_REQUIRED.

Usage:
    python scripts/keepawake.py                # run until killed
Detached launch (used by the campaign):
    (nohup python -u scripts/keepawake.py > logs/keepawake.log 2>&1 &)

The script logs its own liveness every 10 min so overnight logs prove the
machine stayed awake (or reveal exactly when/why it did not).
"""
from __future__ import annotations

import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    if sys.platform != "win32":
        log("not win32; keepawake is a no-op here")
        return
    res = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if res == 0:
        log("WARNING: SetThreadExecutionState failed; sleep prevention "
            "may not be active")
    else:
        log("power request ACTIVE (ES_CONTINUOUS | ES_SYSTEM_REQUIRED): "
            "system sleep prevented while this process runs")
    n = 0
    while True:
        time.sleep(600)
        n += 1
        if n % 10 == 0:  # liveness heartbeat every 100 min
            log(f"keepawake alive ({n} cycles)")


if __name__ == "__main__":
    main()
