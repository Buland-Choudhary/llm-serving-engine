#!/usr/bin/env python3
"""Run a queue of benchmark commands one at a time (one heavy CPU job at a time).

Each non-comment line of the queue file is a shell command. Commands whose ``--out`` file already
exists are skipped, so re-running the queue after an interruption resumes where it stopped.
Progress is appended to ``runs/queue/<queue name>.log``.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    qfile = Path(sys.argv[1])
    log_dir = Path("runs/queue")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{qfile.stem}.log"
    cmds = [ln.strip() for ln in qfile.read_text().splitlines()]
    cmds = [c for c in cmds if c and not c.startswith("#")]
    failures = 0
    for i, cmd in enumerate(cmds, 1):
        m = re.search(r"--out\s+(\S+)", cmd)
        if m and Path(m.group(1)).exists():
            continue
        t0 = time.time()
        with log.open("a") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] ({i}/{len(cmds)}) START {cmd}\n")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        with log.open("a") as fh:
            status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
            fh.write(f"[{time.strftime('%H:%M:%S')}] ({i}/{len(cmds)}) {status} "
                     f"{time.time() - t0:.0f}s\n    " + "\n    ".join(tail) + "\n")
        failures += r.returncode != 0
    with log.open("a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] QUEUE DONE, {failures} failures\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
