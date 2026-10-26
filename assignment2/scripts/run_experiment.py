"""Run the four experiments of Assignment 2 and store every measurement.

Experiments (parameters in config.json):
  1. block_tuning   tiled dense method with several block sizes; the block with the lowest median at the
                    largest tuning size is used for every later "tiled" run (recorded in results/selection.json).
  2. dense          Part A: naive (Assignment 1 baseline), ikj, transpose and tiled over the dense sizes.
  3. sparse_density Part B: n fixed, every structure and density: CSR x CSR (spgemm), dense tiled product of the
                    same matrices (dense), and dense -> CSR conversion cost (convert).
  4. sparse_scaling Part B: fixed structure and density, increasing n, same three modes.

Protocol (same as Assignment 1):
  - every configuration runs in a fresh process pinned to config["cpu_affinity"] with high priority;
  - warm-ups (not reported) and measured repetitions; costly configurations (predicted time of the previous size
    in the same series times the cubic growth > costly_threshold_s) use reps_costly and warmups_costly;
  - a multiplication slower than time_limit_s stops the process ("timeout") and the larger sizes of the series are
    recorded as "not_attempted"; configurations whose predicted memory exceeds memory_budget_mb are not run and are
    recorded as "not_attempted_memory_budget"; a process over the budget is stopped ("memory_exceeded");
  - memory metric: peak working set of the process (Win32 GetProcessMemoryInfo).
Outputs: results/raw_measurements.csv, results/runs.csv, results/environment.json, results/selection.json
"""
import argparse
import csv
import json
import platform
import queue
import statistics
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import psutil
import scipy

from common import (RESULTS, CorePerformanceCounter, build, dense_paths, load_config, peak_working_set_mb,
                    sparse_paths, start_pinned)
from generate_inputs import sparse_configurations

LOAD_MARGIN_S = 120  # extra time before the first line (reading the inputs, building dense copies)
EXPERIMENTS = ["block_tuning", "dense", "sparse_density", "sparse_scaling"]


def on_ac_power():
    battery = psutil.sensors_battery()
    return True if battery is None else bool(battery.power_plugged)


def run_process(args, config):
    """Run one configuration. Returns (lines, status, peak_mb, monitor) where monitor describes the conditions
    of the run: start/end time, whether the laptop stayed on AC power, and the mean CPU utilisation of the whole
    system and of every logical processor that shares the physical core of the pinned one."""
    time.sleep(config.get("cooldown_s", 0))  # same idle period before every configuration
    psutil.cpu_percent(percpu=True)  # reset the utilisation counters
    perf_counter = CorePerformanceCounter(config["cpu_affinity"][0])
    start_time, ac_always = datetime.now(), on_ac_power()
    samples, perf = [], []
    proc = start_pinned(args, config)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:  # the process may already have finished (very short configurations)
        ps = None
    out, status = [], "ok"
    deadline = time.monotonic() + config["time_limit_s"] + LOAD_MARGIN_S
    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            samples.append(psutil.cpu_percent(percpu=True))
            perf.append(perf_counter.sample())
            ac_always &= on_ac_power()
            try:
                over = ps is not None and ps.memory_info().rss / 2**20 > config["memory_budget_mb"]
            except psutil.Error:
                over = False
            if over or time.monotonic() > deadline:
                status = "memory_exceeded" if over else "timeout"
                proc.kill()
                break
            continue
        if line is None:
            break
        out.append(line.split())
        deadline = time.monotonic() + config["time_limit_s"]
    proc.wait()
    peak = peak_working_set_mb(proc)
    if status == "ok" and proc.returncode != 0:
        status = f"error: {proc.stderr.read().strip()[:200]}"
    load = np.array(samples) if samples else np.zeros((1, psutil.cpu_count()))
    sibling = config.get("sibling_cpu")
    monitor = {"start_time": start_time.isoformat(timespec="seconds"),
               "end_time": datetime.now().isoformat(timespec="seconds"), "on_ac_power": ac_always,
               "system_cpu_pct": round(float(load.mean()), 1),
               "sibling_cpu_pct": round(float(load[:, sibling].mean()), 1) if sibling is not None else ""}
    perf = [p for p in perf if p]
    # Effective clock of the pinned core while the configuration ran (sampled every 0.5 s)
    monitor["core_perf_pct"] = round(float(np.mean([p[0] for p in perf])), 1) if perf else ""
    monitor["core_freq_mhz"] = round(float(np.mean([p[0] * p[1] / 100 for p in perf]))) if perf else ""
    monitor["core_freq_min_mhz"] = round(float(min(p[0] * p[1] / 100 for p in perf))) if perf else ""
    return out, status, peak, monitor


def predicted_memory_mb(cfg, nnz_inputs=0):
    """Estimate of the arrays of one configuration (dense float64 matrices and CSR inputs)."""
    n, mode = cfg["n"], cfg["method"]
    dense_bytes = 8 * n * n
    csr_bytes = 12 * nnz_inputs + 16 * (n + 1)
    if mode == "transpose":
        return 4 * dense_bytes / 2**20
    if mode in ("naive", "ikj", "tiled"):
        return 3 * dense_bytes / 2**20
    if mode == "dense":
        return (3 * dense_bytes + csr_bytes) / 2**20
    if mode == "convert":
        return (2 * dense_bytes + 2 * csr_bytes) / 2**20
    return 3 * csr_bytes / 2**20  # spgemm: inputs plus a result of comparable size (estimate only)


def configurations(config, block):
    d, sp = config["dense"], config["sparse"]
    manifest = {}
    with open(RESULTS / "inputs_manifest.csv") as f:
        for row in csv.DictReader(f):
            manifest[row["file"]] = int(row["nnz"])

    def sparse_cfgs(experiment, n, structure, density):
        a, b = sparse_paths(config, n, structure, density)
        nnz = manifest[a.name] + manifest[b.name]
        for mode in ("spgemm", "dense", "convert"):
            yield {"experiment": experiment, "method": mode, "n": n, "block": block if mode == "dense" else "",
                   "structure": structure, "density_pct": density, "nnz_inputs": nnz,
                   # scaling: one series per mode over n; density sweep: every configuration is its own series
                   "series": (experiment, mode) if experiment == "sparse_scaling" else (experiment, mode, structure, density),
                   "args": [mode] + ([str(a), str(b), str(block)] if mode == "dense" else [str(a), str(b)])}

    out = {"block_tuning": [], "dense": [], "sparse_density": [], "sparse_scaling": []}
    for n in d["block_tuning"]["sizes"]:
        for b in d["block_tuning"]["blocks"]:
            out["block_tuning"].append({"experiment": "block_tuning", "method": "tiled", "n": n, "block": b,
                                        "structure": "dense", "density_pct": 100, "nnz_inputs": 2 * n * n,
                                        "series": ("block_tuning", b), "dense": True})
    for m in d["methods"]:
        for n in d["sizes"]:
            out["dense"].append({"experiment": "dense", "method": m, "n": n, "block": block if m == "tiled" else "",
                                 "structure": "dense", "density_pct": 100, "nnz_inputs": 2 * n * n,
                                 "series": ("dense", m), "dense": True})
    for structure in sp["structures"]:
        for density in sorted(sp["densities_pct"], reverse=True):
            out["sparse_density"].extend(sparse_cfgs("sparse_density", sp["n"], structure, density))
    sc = sp["scaling"]
    for n in sc["sizes"]:
        out["sparse_scaling"].extend(sparse_cfgs("sparse_scaling", n, sc["structure"], sc["density_pct"]))
    for e in out:  # run every series in increasing n, so that costly and stop rules can use the previous size
        out[e].sort(key=lambda c: (str(c["series"]), c["n"]))
    return out


def environment(config, block):
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"date": datetime.now().isoformat(timespec="seconds"), "cpu": cpu,
            "physical_cores": psutil.cpu_count(logical=False), "logical_processors": psutil.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python": platform.python_version(), "numpy": np.__version__, "scipy (reference only)": scipy.__version__,
            "gcc": gcc, "c_flags": config["c_flags"], "cpu_affinity": config["cpu_affinity"],
            "priority": config["priority"], "power_plugged": getattr(psutil.sensors_battery(), "power_plugged", None),
            "selected_block": block, "config": config}


def save(rows, name):
    if rows:
        with open(RESULTS / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS)
    args = parser.parse_args()

    if config.get("require_ac_power") and not on_ac_power():
        raise SystemExit("The laptop is running on battery: connect the charger before benchmarking.")
    programs = build(config)
    RESULTS.mkdir(exist_ok=True)
    raw, runs = [], []
    # When only some experiments are run, keep the stored rows of the other experiments
    for name, rows in (("raw_measurements.csv", raw), ("runs.csv", runs)):
        if (RESULTS / name).exists():
            with open(RESULTS / name) as f:
                rows.extend(r for r in csv.DictReader(f) if r["experiment"] not in args.experiments)
    selection_path = RESULTS / "selection.json"
    block = config["dense"]["block"]
    if selection_path.exists():
        block = json.loads(selection_path.read_text())["block"]

    for experiment in args.experiments:
        if experiment != "block_tuning" and "block_tuning" in args.experiments:
            block = json.loads(selection_path.read_text())["block"]
        cfgs = configurations(config, block)[experiment]
        prev = {}  # series -> (n, median) of the last successful configuration
        stopped = set()
        for cfg in cfgs:
            series, n = cfg["series"], cfg["n"]
            mem = predicted_memory_mb(cfg, cfg["nnz_inputs"] if cfg["method"] in ("spgemm", "dense", "convert") else 0)
            label = {k: cfg[k] for k in ("experiment", "method", "n", "block", "structure", "density_pct")}
            if series in stopped or mem > config["memory_budget_mb"]:
                status = "not_attempted" if series in stopped else "not_attempted_memory_budget"
                runs.append({**label, "status": status, "warmups": 0, "reps": 0, "predicted_memory_mb": round(mem, 1),
                             "peak_working_set_mb": ""})
                print(f"{experiment:14s} {cfg['method']:9s} n={n:6d} {cfg['structure']:8s} d={cfg['density_pct']:<6g} {status}")
                if status == "not_attempted_memory_budget":
                    stopped.add(series)
                save(runs, "runs.csv")
                continue
            costly = series in prev and prev[series][1] * (n / prev[series][0]) ** 3 > config["costly_threshold_s"]
            warmups = config["warmups_costly" if costly else "warmups"]
            reps = config["reps_costly" if costly else "reps"]
            if cfg.get("dense"):
                a, b = dense_paths(config, n)
                cmd = [programs["dense"], "bench", cfg["method"], str(n), str(warmups), str(reps), str(a.parent),
                       str(cfg["block"] or 64)]
            else:
                cmd = [programs["sparse"], cfg["args"][0], str(warmups), str(reps), *cfg["args"][1:]]
            lines, status, peak, monitor = run_process(cmd, config)

            for i, parts in enumerate(lines):
                kind, seconds, checksum = parts[0], float(parts[1]), int(parts[2])
                raw.append({**label, "dtype": "float64", "kind": kind, "index": i if kind == "warmup" else i - warmups,
                            "time_s": seconds, "checksum_bits": checksum,
                            "nnz_out": int(parts[3]) if len(parts) > 3 else "", "threads": 1,
                            "cpu_affinity": " ".join(map(str, config["cpu_affinity"]))})
            times = [float(p[1]) for p in lines if p[0] == "rep"]
            runs.append({**label, "status": status, "warmups": warmups, "reps": len(times),
                         "predicted_memory_mb": round(mem, 1), "peak_working_set_mb": f"{peak:.1f}" if peak else "",
                         **monitor})
            if not monitor["on_ac_power"]:
                print("WARNING: the laptop was not on AC power during this configuration")
            if status == "ok" and times:
                prev[series] = (n, statistics.median(times))
                print(f"{experiment:14s} {cfg['method']:9s} n={n:6d} {cfg['structure']:8s} d={cfg['density_pct']:<6g}"
                      f" b={cfg['block'] or '-':<4} median={prev[series][1]:10.5f} s reps={len(times)} peak={peak:8.1f} MB")
            else:
                print(f"{experiment:14s} {cfg['method']:9s} n={n:6d} {status}")
                stopped.add(series)
            save(raw, "raw_measurements.csv")
            save(runs, "runs.csv")

        if experiment == "block_tuning":
            tun = [r for r in raw if r["experiment"] == "block_tuning" and r["kind"] == "rep"]
            n_max = max(r["n"] for r in tun)
            medians = {b: statistics.median(r["time_s"] for r in tun if r["n"] == n_max and r["block"] == b)
                       for b in config["dense"]["block_tuning"]["blocks"]}
            block = min(medians, key=medians.get)
            selection_path.write_text(json.dumps({"block": block, "n": n_max, "medians_s": medians}, indent=2))
            print(f"Selected block size: {block} (medians at n={n_max}: {medians})")

    (RESULTS / "environment.json").write_text(json.dumps(environment(config, block), indent=2))
    print(f"\nResults saved in {RESULTS}")


if __name__ == "__main__":
    main()
