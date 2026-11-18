"""Summarise the Assignment 3 measurements and produce the tables and figures of the report.

Speedup S(p) = T_seq(n) / T_p(n) and efficiency E(p) = S(p) / p, where T_seq is the median time of the sequential
blocked method and T_p the median time of the parallel method with p threads, for the same n (medians of the
measured repetitions). The parallel method with p = 1 is compared with T_seq to show the parallel overhead.
Quartiles use numpy.percentile with linear interpolation.

Outputs: results/summary.csv, results/speedup.csv, results/schedule.csv and
         results/fig1_speedup.png, fig2_efficiency.png, fig3_time.png, fig4_schedule.png, fig5_memory.png
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, load_config

SIZE_STYLE = {1024: ("#3b6fb6", "o"), 2048: ("#d9822b", "s"), 4096: ("#2a9d5c", "^")}
REGIONS = [(0.5, 6.5, "#e8eef7", "physical\nP-cores"), (6.5, 12.5, "#f7efe6", "+ hyper-\nthreads"),
           (12.5, 22.5, "#eaf4ee", "+ E-cores"), (22.5, 50, "#f2f2f2", "over-\nsubscribed")]


def shade(ax, ymax):
    for lo, hi, color, label in REGIONS:
        ax.axvspan(lo, hi, color=color, zorder=0)
        ax.text(np.sqrt(lo * hi), ymax * 0.97, label, ha="center", va="top", fontsize=8, color="#555555")


def main():
    config = load_config()
    raw = pd.read_csv(RESULTS / "raw_measurements.csv")
    runs = pd.read_csv(RESULTS / "runs.csv")
    keys = ["experiment", "method", "n", "threads", "schedule"]
    reps = raw[raw["kind"] == "rep"]
    stats = reps.groupby(keys).agg(
        median=("time_s", "median"), q1=("time_s", lambda x: np.percentile(x, 25)),
        q3=("time_s", lambda x: np.percentile(x, 75)), min=("time_s", "min"), max=("time_s", "max"),
        measured_reps=("time_s", "count"), team_size=("team_size", "first"),
        checksums=("checksum_bits", "nunique")).reset_index()
    stats["iqr_pct_of_median"] = 100 * (stats["q3"] - stats["q1"]) / stats["median"]
    summary = runs.merge(stats, on=keys, how="left")
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.6g")

    # Checksums: every repetition and every thread count of a size must give the same result
    same_result = reps.groupby("n")["checksum_bits"].nunique()

    seq = summary[summary["method"] == "seq"].set_index("n")["median"]
    par = summary[(summary["experiment"] == "scaling") & (summary["method"] == "par")].copy()
    par["T_seq"] = par["n"].map(seq)
    par["speedup"] = par["T_seq"] / par["median"]
    par["efficiency"] = par["speedup"] / par["threads"]
    par["speedup_q_low"] = par["T_seq"] / par["q3"]
    par["speedup_q_high"] = par["T_seq"] / par["q1"]
    par["overhead_vs_seq"] = np.where(par["threads"] == 1, par["median"] / par["T_seq"], np.nan)
    par.to_csv(RESULTS / "speedup.csv", index=False, float_format="%.6g")

    sched = summary[(summary["method"] == "par") & (summary["n"] == config["schedule_comparison"]["n"]) &
                    (summary["threads"].isin(config["schedule_comparison"]["threads"]))]
    sched = sched.pivot_table(index="threads", columns="schedule", values="median").reset_index()
    sched["static_over_dynamic"] = sched["static"] / sched["dynamic"]
    sched.to_csv(RESULTS / "schedule.csv", index=False, float_format="%.6g")

    threads = sorted(par["threads"].unique())

    # Figure 1: speedup
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ymax = max(par["speedup_q_high"].max() * 1.15, 8)
    shade(ax, ymax)
    ax.plot([1, 22], [1, 22], "k--", lw=0.8, label="ideal S(p) = p")
    for n, g in par.groupby("n"):
        color, marker = SIZE_STYLE[n]
        g = g.sort_values("threads")
        ax.errorbar(g["threads"], g["speedup"], yerr=[g["speedup"] - g["speedup_q_low"], g["speedup_q_high"] - g["speedup"]],
                    marker=marker, color=color, capsize=3, label=f"n = {n}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Threads p")
    ax.set_ylabel("Speedup S(p) = T_seq / T_p (medians)")
    ax.set_title("Speedup of the parallel blocked multiplication (bars: from the IQR of T_p)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_speedup.png", dpi=170)

    # Figure 2: efficiency
    fig, ax = plt.subplots(figsize=(9, 4.8))
    shade(ax, 1.25)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    for n, g in par.groupby("n"):
        color, marker = SIZE_STYLE[n]
        g = g.sort_values("threads")
        ax.plot(g["threads"], g["efficiency"], marker=marker, color=color, label=f"n = {n}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Threads p")
    ax.set_ylabel("Efficiency E(p) = S(p) / p")
    ax.set_title("Parallel efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_efficiency.png", dpi=170)

    # Figure 3: time
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for n, g in par.groupby("n"):
        color, marker = SIZE_STYLE[n]
        g = g.sort_values("threads")
        ax.errorbar(g["threads"], g["median"], yerr=[g["median"] - g["q1"], g["q3"] - g["median"]], marker=marker,
                    color=color, capsize=3, label=f"parallel, n = {n}")
        ax.axhline(seq[n], color=color, ls=":", lw=1, label=f"sequential, n = {n}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    ax.set_xlabel("Threads p")
    ax.set_ylabel("Median time per multiplication (s)")
    ax.set_title("Execution time versus threads (dotted: sequential baseline)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_time.png", dpi=170)

    # Figure 4: static vs dynamic schedule
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(sched))
    ax.bar(x - 0.2, sched["dynamic"], width=0.4, color="#3b6fb6", label="dynamic (1 block row at a time)")
    ax.bar(x + 0.2, sched["static"], width=0.4, color="#d9822b", label="static (equal shares)")
    for xi, d, s in zip(x, sched["dynamic"], sched["static"]):
        ax.text(xi - 0.2, d, f"{d:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p = {p}" for p in sched["threads"]])
    ax.set_ylabel("Median time (s)")
    ax.set_title(f"Loop schedule, n = {config['schedule_comparison']['n']}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_schedule.png", dpi=170)

    # Figure 5: memory
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for n, g in par.groupby("n"):
        color, marker = SIZE_STYLE[n]
        g = g.sort_values("threads")
        ax.plot(g["threads"], g["peak_working_set_mb"], marker=marker, color=color, label=f"parallel, n = {n}")
        seq_mem = summary[(summary["method"] == "seq") & (summary["n"] == n)]["peak_working_set_mb"].iloc[0]
        ax.axhline(seq_mem, color=color, ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    ax.set_xlabel("Threads p")
    ax.set_ylabel("Peak working set (MiB)")
    ax.set_title("Memory versus threads (dotted: sequential)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_memory.png", dpi=170)

    pd.set_option("display.width", 220)
    print("Sequential medians:", seq.round(4).to_dict())
    print(par[["n", "threads", "logical_cpus", "team_size", "median", "iqr_pct_of_median", "speedup", "efficiency",
               "overhead_vs_seq", "peak_working_set_mb", "on_ac_power", "other_cpus_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(sched.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("Sizes whose results differ between runs or thread counts:", int((same_result > 1).sum()))
    print("Configurations not on AC power:", int((~summary["on_ac_power"].astype(bool)).sum()), "of", len(summary))


if __name__ == "__main__":
    main()
