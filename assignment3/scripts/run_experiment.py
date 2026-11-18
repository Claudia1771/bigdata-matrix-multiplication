"""Run the Assignment 3 experiment: sequential baseline vs OpenMP parallel version for several thread counts.

Configurations (config.json):
  - for every size: the sequential blocked method (T_seq, pinned to one performance core) and the parallel method
    with every thread count in config["threads"] (dynamic schedule);
  - schedule comparison: the parallel method with the static schedule for config["schedule_comparison"].
Affinity (config["affinity_plan"]): p <= 6 one thread per physical performance core; 7-12 the hyper-thread siblings
are added; > 12 all logical processors (efficient and low-power cores included); 44 threads = 2x oversubscription.
Protocol as in Assignments 1-2: fresh process per configuration, warm-ups, repetitions, median and IQR, 300 s and
4 GiB budgets, peak working set, AC power required, idle period before every configuration, and the utilisation of
the system and of the allowed logical processors recorded for every configuration.
Outputs: results/raw_measurements.csv, results/runs.csv, results/environment.json
"""
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

from common import RESULTS, affinity_mask, build, data_dir, load_config, peak_working_set_mb, start_with_affinity

LOAD_MARGIN_S = 60


def on_ac_power():
    battery = psutil.sensors_battery()
    return True if battery is None else bool(battery.power_plugged)


def run_process(args, mask, config):
    time.sleep(config.get("cooldown_s", 0))
    psutil.cpu_percent(percpu=True)
    start, ac_always, samples = datetime.now(), on_ac_power(), []
    proc = start_with_affinity(args, mask, config)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    out, status = [], "ok"
    deadline = time.monotonic() + config["time_limit_s"] + LOAD_MARGIN_S
    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            samples.append(psutil.cpu_percent(percpu=True))
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
    others = [c for c in range(load.shape[1]) if c not in mask]
    monitor = {"start_time": start.isoformat(timespec="seconds"), "end_time": datetime.now().isoformat(timespec="seconds"),
               "on_ac_power": ac_always,
               "allowed_cpus_pct": round(float(load[:, mask].mean()), 1),
               "other_cpus_pct": round(float(load[:, others].mean()), 1) if others else ""}
    return out, status, peak, monitor


def configurations(config):
    cfgs = []
    for n in config["sizes"]:
        cfgs.append({"experiment": "scaling", "method": "seq", "n": n, "threads": 1, "schedule": "-"})
        for p in config["threads"]:
            cfgs.append({"experiment": "scaling", "method": "par", "n": n, "threads": p, "schedule": config["schedule"]})
    sc = config["schedule_comparison"]
    for p in sc["threads"]:
        cfgs.append({"experiment": "schedule", "method": "par", "n": sc["n"], "threads": p, "schedule": "static"})
    return cfgs


def environment(config, exe):
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"date": datetime.now().isoformat(timespec="seconds"), "cpu": cpu,
            "physical_cores": psutil.cpu_count(logical=False), "logical_processors": psutil.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "gcc": gcc, "c_flags": config["c_flags"], "openmp": "libgomp (GCC), OMP_DYNAMIC=false",
            "priority": config["priority"], "config": config}


def save(rows, name):
    if rows:
        with open(RESULTS / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    config = load_config()
    allow_battery = "--allow-battery" in sys.argv
    if config.get("require_ac_power") and not on_ac_power() and not allow_battery:
        raise SystemExit("The laptop is running on battery: connect the charger, or pass --allow-battery to run anyway "
                         "(the power state of every configuration is recorded in runs.csv).")
    exe = build(config)
    RESULTS.mkdir(exist_ok=True)
    env = environment(config, exe)
    env["power_at_start"] = "AC" if on_ac_power() else "battery"
    env["allow_battery"] = allow_battery
    env["power_mode_overlay_ac"] = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes')."
         "ActiveOverlayAcPowerScheme, (Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\"
         "PowerSchemes').ActiveOverlayDcPowerScheme"], capture_output=True, text=True).stdout.split()
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2))
    data = data_dir(config)
    raw, runs, prev = [], [], {}

    for cfg in configurations(config):
        key = (cfg["experiment"], cfg["method"], cfg["threads"], cfg["schedule"])
        costly = key in prev and prev[key][1] * (cfg["n"] / prev[key][0]) ** 3 > config["costly_threshold_s"]
        warmups = config["warmups_costly" if costly else "warmups"]
        reps = config["reps_costly" if costly else "reps"]
        mask = affinity_mask(config, cfg["method"], cfg["threads"])
        args = [exe, "bench", cfg["method"], str(cfg["n"]), str(cfg["threads"]), str(warmups), str(reps), str(data),
                str(config["block"]), cfg["schedule"] if cfg["schedule"] != "-" else "dynamic"]
        lines, status, peak, monitor = run_process(args, mask, config)

        label = {**cfg, "affinity": " ".join(map(str, mask)), "logical_cpus": len(mask)}
        for i, parts in enumerate(lines):
            raw.append({**label, "dtype": "float64", "kind": parts[0], "index": i if parts[0] == "warmup" else i - warmups,
                        "time_s": float(parts[1]), "checksum_bits": int(parts[2]), "team_size": int(parts[3])})
        times = [float(p[1]) for p in lines if p[0] == "rep"]
        runs.append({**label, "status": status, "warmups": warmups, "reps": len(times),
                     "peak_working_set_mb": f"{peak:.1f}" if peak else "", **monitor})
        if status == "ok" and times:
            prev[key] = (cfg["n"], statistics.median(times))
            print(f"{cfg['experiment']:8s} {cfg['method']} n={cfg['n']:5d} p={cfg['threads']:2d} {cfg['schedule']:7s} "
                  f"median={prev[key][1]:9.4f} s reps={len(times)} cpus={len(mask):2d} other_load={monitor['other_cpus_pct']}",
                  flush=True)
        else:
            print(f"{cfg['experiment']:8s} {cfg['method']} n={cfg['n']} p={cfg['threads']} {status}", flush=True)
        if not monitor["on_ac_power"]:
            print("WARNING: the laptop was not on AC power during this configuration", flush=True)
        save(raw, "raw_measurements.csv")
        save(runs, "runs.csv")
    print(f"\nResults saved in {RESULTS}")


if __name__ == "__main__":
    main()
