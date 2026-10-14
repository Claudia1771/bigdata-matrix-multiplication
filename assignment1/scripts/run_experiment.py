"""Run the Assignment 1 benchmark (Python, Java and C basic triple loop).

Protocol (parameters in config.json):
  - For each language, sizes are run in increasing order. Each configuration runs in a fresh process
    that loads the shared inputs, performs the warm-up multiplications (not reported) and then the
    measured repetitions. Every measurement is stored in results/raw_measurements.csv.
  - Costly configurations: if the median of the previous size multiplied by 8 (cubic model) is above
    costly_threshold_s, the configuration uses reps_costly and warmups_costly.
  - Time limit: if a single multiplication takes longer than time_limit_s, the process is stopped and
    the configuration is recorded as "timeout"; larger sizes are then recorded as "not_attempted".
  - Memory budget: if the process working set exceeds memory_budget_mb it is stopped and recorded as
    "memory_exceeded".
  - Affinity and priority: every benchmark process is started pinned to the logical processors in
    cpu_affinity (a performance core of the hybrid CPU, see core_probe.py) with high priority.
  - Memory metric: peak working set (peak resident memory) of the whole process, read with the Win32
    GetProcessMemoryInfo API. Stored in results/memory.csv.
Usage: python run_experiment.py [--sizes 64 128 ...] [--languages Python Java C]
"""
import argparse
import csv
import json
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import psutil

from common import DATA, LANGUAGES, RESULTS, build, load_config, peak_working_set_mb

LOAD_MARGIN_S = 60  # extra time allowed before the first line (input loading, JVM start-up)


def start_pinned(args, config):
    """Start a process already pinned to config["cpu_affinity"] and with the configured priority.

    Windows child processes inherit the affinity mask of their parent, so the harness pins itself
    while creating the child and then restores its own affinity. This way the program (and the JVM,
    which reads the number of available processors at start-up) never runs on another core.
    """
    me = psutil.Process()
    original = me.cpu_affinity()
    flags = subprocess.HIGH_PRIORITY_CLASS if config.get("priority") == "high" and sys.platform == "win32" else 0
    me.cpu_affinity(config["cpu_affinity"])
    try:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags)
    finally:
        me.cpu_affinity(original)


def run_configuration(cmd, n, warmups, reps, config):
    """Run one (language, n) configuration. Returns (measurements, status, peak_mb)."""
    proc = start_pinned(cmd + ["bench", str(n), str(warmups), str(reps), str(DATA)], config)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()
    ps = psutil.Process(proc.pid)

    measurements, status = [], "ok"
    deadline = time.monotonic() + config["time_limit_s"] + LOAD_MARGIN_S
    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            try:
                over_memory = ps.memory_info().rss / 2**20 > config["memory_budget_mb"]
            except psutil.Error:
                over_memory = False
            if over_memory or time.monotonic() > deadline:
                status = "memory_exceeded" if over_memory else "timeout"
                proc.kill()
                break
            continue
        if line is None:
            break
        kind, seconds, checksum = line.split()
        measurements.append((kind, float(seconds), int(checksum)))
        deadline = time.monotonic() + config["time_limit_s"]

    proc.wait()
    peak = peak_working_set_mb(proc)
    if status == "ok" and proc.returncode != 0:
        status = f"error: {proc.stderr.read().strip()[:200]}"
    return measurements, status, peak


def environment(config):
    def first_line(cmd, stream="stdout"):
        out = subprocess.run(cmd, capture_output=True, text=True)
        return (out.stdout if stream == "stdout" else out.stderr).strip().splitlines()[0]

    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "cpu": cpu or platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "numpy (input generation and reference only)": np.__version__,
        "java": first_line(["java", "-version"], "stderr"),
        "java_flags": config["java_flags"],
        "gcc": first_line(["gcc", "--version"]),
        "c_flags": config["c_flags"],
        "cpu_affinity": config["cpu_affinity"],
        "priority": config.get("priority", "normal"),
        "power_plugged": getattr(psutil.sensors_battery(), "power_plugged", None),
        "config": config,
    }


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=config["sizes"])
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    args = parser.parse_args()

    missing = [n for n in args.sizes if not (DATA / f"A_{n}.bin").exists()]
    if missing:
        sys.exit(f"Missing input files for n={missing}. Run generate_inputs.py first.")

    commands = build(config)
    RESULTS.mkdir(exist_ok=True)
    env = environment(config)
    with open(RESULTS / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(json.dumps({k: v for k, v in env.items() if k != "config"}, indent=2), "\n")

    raw_rows, mem_rows = [], []
    for lang in args.languages:
        previous_median, stop = None, False
        for n in sorted(args.sizes):
            if stop:
                mem_rows.append({"language": lang, "n": n, "status": "not_attempted", "warmups": 0,
                                 "reps": 0, "peak_working_set_mb": ""})
                print(f"{lang:6s} n={n:5d}  not attempted (smaller size exceeded the budget)")
                continue
            costly = previous_median is not None and previous_median * (n / prev_n) ** 3 > config["costly_threshold_s"]
            warmups = config["warmups_costly" if costly else "warmups"][lang]
            reps = config["reps_costly" if costly else "reps"]

            measurements, status, peak = run_configuration(commands[lang], n, warmups, reps, config)
            for i, (kind, seconds, checksum) in enumerate(measurements):
                raw_rows.append({"language": lang, "n": n, "dtype": config["dtype"], "kind": kind,
                                 "index": i if kind == "warmup" else i - warmups,
                                 "time_s": seconds, "checksum_bits": checksum, "threads": 1,
                                 "cpu_affinity": " ".join(map(str, config["cpu_affinity"]))})
            mem_rows.append({"language": lang, "n": n, "status": status, "warmups": warmups,
                             "reps": sum(1 for m in measurements if m[0] == "rep"),
                             "peak_working_set_mb": f"{peak:.1f}" if peak else ""})

            times = [s for k, s, _ in measurements if k == "rep"]
            if status == "ok" and times:
                previous_median, prev_n = statistics.median(times), n
                print(f"{lang:6s} n={n:5d}  median={previous_median:10.4f} s  reps={len(times)} "
                      f"warmups={warmups}  peak={peak:7.1f} MB")
            else:
                print(f"{lang:6s} n={n:5d}  {status} after {len(times)} measured reps (limit {config['time_limit_s']} s)")
                stop = True

            # Save after every configuration so that no data is lost if the run is interrupted
            for name, rows in (("raw_measurements.csv", raw_rows), ("memory.csv", mem_rows)):
                if rows:
                    with open(RESULTS / name, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)

    print(f"\nRaw results saved in {RESULTS}")


if __name__ == "__main__":
    main()
