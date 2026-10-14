"""Measure the single-thread speed of every logical processor.

The Intel Core Ultra 7 155H is a hybrid processor (performance, efficient and low-power efficient cores).
This script runs the same small CPU-bound Python loop pinned to each logical processor, so that the
benchmark can be pinned to a performance core. Output: results/core_probe.csv
"""
import csv
import subprocess
import sys

import psutil

from common import RESULTS

CODE = "import time\nt=time.perf_counter()\ns=0\nfor i in range(3_000_000): s+=i*i\nprint(time.perf_counter()-t)"


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for cpu in range(psutil.cpu_count()):
        times = []
        for _ in range(3):
            proc = subprocess.Popen([sys.executable, "-c", CODE], stdout=subprocess.PIPE, text=True)
            psutil.Process(proc.pid).cpu_affinity([cpu])
            times.append(float(proc.communicate()[0]))
        rows.append({"logical_cpu": cpu, "median_s": sorted(times)[1], "min_s": min(times), "max_s": max(times)})
        print(f"CPU {cpu:2d}: median {rows[-1]['median_s']:.3f} s")
    with open(RESULTS / "core_probe.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
