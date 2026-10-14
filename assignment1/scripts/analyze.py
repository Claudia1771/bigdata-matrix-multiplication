"""Summarise the raw measurements and produce the tables and figures of the report.

Inputs:  results/raw_measurements.csv, results/memory.csv, config.json
Outputs: results/summary.csv            median, quartiles, IQR, min, max, peak memory, growth ratios
         results/fit.csv                fitted exponent b of time = a * n^b per language
         results/fig1_time_vs_n.png     median time vs n (log-log) with IQR bars
         results/fig2_time_per_op.png   median time / n^3 (ns per multiply-add)
         results/fig3_memory.png        peak working set vs n, with the array-storage estimate
         results/fig4_dispersion.png    every measured repetition relative to its median
         results/pinned_vs_unpinned.csv comparison with the preliminary run without CPU affinity
         results/fig5_pinned_vs_unpinned.png  the same comparison as a figure
Quartiles use numpy.percentile with linear interpolation.
"""
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import LANGUAGES, RESULTS, load_config

COLORS = {"Python": "#3b6fb6", "Java": "#d9822b", "C": "#2a9d5c"}
MARKERS = {"Python": "o", "Java": "s", "C": "^"}


def set_size_ticks(ax, summary):
    """Label only the power-of-two sizes (n = 1000 is drawn just left of 1024 without its own label)."""
    ticks = [n for n in sorted(summary["n"].unique()) if n & (n - 1) == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def summarise(raw, mem):
    reps = raw[raw["kind"] == "rep"]
    stats = reps.groupby(["language", "n"])["time_s"].agg(
        median="median", q1=lambda x: np.percentile(x, 25), q3=lambda x: np.percentile(x, 75),
        min="min", max="max", measured_reps="count").reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["iqr_pct_of_median"] = 100 * stats["iqr"] / stats["median"]
    stats["ns_per_multiply_add"] = 1e9 * stats["median"] / stats["n"].astype(float) ** 3

    summary = mem.merge(stats, on=["language", "n"], how="left")
    summary = summary.sort_values(["language", "n"], key=lambda c: c.map(LANGUAGES.index) if c.name == "language" else c)

    # Growth between consecutive successful sizes, compared with the cubic model (n_i / n_{i-1})^3
    for lang, g in summary[summary["status"] == "ok"].groupby("language"):
        g = g.sort_values("n")
        summary.loc[g.index[1:], "observed_ratio"] = g["median"].values[1:] / g["median"].values[:-1]
        summary.loc[g.index[1:], "cubic_ratio"] = (g["n"].values[1:] / g["n"].values[:-1]) ** 3

    # Speed relative to C at the same size (median time / median C time)
    c_times = summary[summary["language"] == "C"].set_index("n")["median"]
    summary["time_relative_to_C"] = summary["median"] / summary["n"].map(c_times)

    # Checksum consistency: every measurement of a size must produce the same result in all languages
    checks = raw.groupby("n")["checksum_bits"].nunique()
    summary["checksum_consistent"] = summary["n"].map(checks == 1)
    return summary


def fit_exponents(summary, min_n=256):
    rows = []
    for lang, g in summary[(summary["status"] == "ok") & (summary["n"] >= min_n)].groupby("language"):
        b, log_a = np.polyfit(np.log(g["n"]), np.log(g["median"]), 1)
        rows.append({"language": lang, "sizes": " ".join(map(str, g["n"])), "exponent_b": round(b, 3),
                     "coefficient_a": np.exp(log_a)})
    return pd.DataFrame(rows)


def plot(summary, config):
    ok = summary[summary["status"] == "ok"]

    # Figure 1: median time vs n with IQR bars; failed configurations drawn at the time limit
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for lang in LANGUAGES:
        g = ok[ok["language"] == lang]
        ax.errorbar(g["n"], g["median"], yerr=[g["median"] - g["q1"], g["q3"] - g["median"]],
                    marker=MARKERS[lang], color=COLORS[lang], capsize=3, label=lang)
        failed = summary[(summary["language"] == lang) & (summary["status"].isin(["timeout", "memory_exceeded"]))]
        if len(failed):
            ax.scatter(failed["n"], [config["time_limit_s"]] * len(failed), marker="x", s=80, color=COLORS[lang],
                       label=f"{lang} timeout (> {config['time_limit_s']} s)", zorder=5)
    n_ref = np.array([ok["n"].min(), ok["n"].max()], dtype=float)
    anchor = ok[(ok["language"] == "C") & (ok["n"] == 512)]["median"].iloc[0]
    ax.plot(n_ref, anchor * (n_ref / 512) ** 3, "k--", lw=1, label="n³ reference (slope 3)")
    ax.axhline(config["time_limit_s"], color="grey", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    set_size_ticks(ax, summary)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Execution time per multiplication (s)")
    ax.set_title("Median execution time (bars: interquartile range)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_time_vs_n.png", dpi=170)

    # Figure 2: time per multiply-add (median / n^3)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lang in LANGUAGES:
        g = ok[ok["language"] == lang]
        ax.plot(g["n"], g["ns_per_multiply_add"], marker=MARKERS[lang], color=COLORS[lang], label=lang)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    set_size_ticks(ax, summary)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Median time / n³ (ns per multiply-add)")
    ax.set_title("Cost per inner-loop iteration (constant if time grows exactly as n³)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_time_per_op.png", dpi=170)

    # Figure 3: peak working set vs n, with the storage estimate of three n x n float64 matrices
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lang in LANGUAGES:
        g = summary[(summary["language"] == lang) & summary["peak_working_set_mb"].notna()]
        ax.plot(g["n"], g["peak_working_set_mb"], marker=MARKERS[lang], color=COLORS[lang], label=lang)
    n_all = np.array(sorted(summary["n"].unique()), dtype=float)
    ax.plot(n_all, 3 * 8 * n_all ** 2 / 2**20, "k--", lw=1, label="Estimate: 3 matrices × 8n² bytes")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    set_size_ticks(ax, summary)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Peak working set of the process (MiB)")
    ax.set_title("Peak memory (whole process, including the runtime)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_memory.png", dpi=170)


def plot_dispersion(raw):
    reps = raw[raw["kind"] == "rep"].copy()
    reps["relative"] = reps["time_s"] / reps.groupby(["language", "n"])["time_s"].transform("median")
    sizes = sorted(reps["n"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for j, lang in enumerate(LANGUAGES):
        g = reps[reps["language"] == lang]
        x = np.array([sizes.index(n) for n in g["n"]]) + (j - 1) * width
        ax.scatter(x, g["relative"], color=COLORS[lang], marker=MARKERS[lang], s=18, alpha=0.8, label=lang)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Measured time / median of its configuration")
    ax.set_title("Dispersion of the individual repetitions")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_dispersion.png", dpi=170)


def compare_preliminary(summary):
    """Compare the reported (pinned) run with the preliminary run without CPU affinity, if available."""
    path = RESULTS / "preliminary_run_unpinned" / "raw_measurements.csv"
    if not path.exists():
        return None
    pre = pd.read_csv(path)
    pre = pre[pre["kind"] == "rep"].groupby(["language", "n"])["time_s"].agg(
        unpinned_median="median", unpinned_min="min", unpinned_max="max",
        unpinned_iqr=lambda x: np.percentile(x, 75) - np.percentile(x, 25)).reset_index()
    pre["unpinned_max_over_min"] = pre["unpinned_max"] / pre["unpinned_min"]
    cur = summary[summary["status"] == "ok"][["language", "n", "median", "min", "max", "iqr"]].copy()
    cur["pinned_max_over_min"] = cur["max"] / cur["min"]
    both = cur.merge(pre, on=["language", "n"], how="outer")
    both.to_csv(RESULTS / "pinned_vs_unpinned.csv", index=False, float_format="%.6g")
    plot_pinned_vs_unpinned(both)
    return both


def plot_pinned_vs_unpinned(both, n_ref=1024):
    """(a) median at n_ref with min-max range for both runs; (b) max/min ratio of every configuration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ref = both[both["n"] == n_ref].set_index("language").loc[LANGUAGES]
    x = np.arange(len(LANGUAGES))
    for offset, prefix, label, hatch in ((-0.2, "unpinned_", "Preliminary (no affinity)", "//"),
                                         (0.2, "", "Final (pinned to a P-core)", None)):
        med = ref[f"{prefix}median"]
        err = [med - ref[f"{prefix}min"], ref[f"{prefix}max"] - med]
        ax1.bar(x + offset, med, width=0.4, yerr=err, capsize=4, label=label,
                color=[COLORS[l] for l in LANGUAGES], alpha=0.45 if hatch else 0.95, hatch=hatch,
                edgecolor="black", linewidth=0.6)
        for xi, m, top in zip(x + offset, med, ref[f"{prefix}max"]):
            ax1.text(xi, top * 1.08, f"{m:.3g} s", ha="center", va="bottom", fontsize=7.5)
    ax1.set_yscale("log")
    ax1.set_ylim(top=ref[["max", "unpinned_max"]].max().max() * 4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LANGUAGES)
    ax1.set_ylabel(f"Time per multiplication, n = {n_ref} (s)")
    ax1.set_title(f"(a) Median and min–max range, n = {n_ref}")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    for j, lang in enumerate(LANGUAGES):
        g = both[both["language"] == lang]
        ax2.scatter(np.full(len(g), j - 0.12), g["unpinned_max_over_min"], marker="x", color=COLORS[lang], s=35,
                    label="Preliminary (no affinity)" if j == 0 else None)
        ax2.scatter(np.full(len(g), j + 0.12), g["pinned_max_over_min"], marker="o", color=COLORS[lang], s=25,
                    label="Final (pinned)" if j == 0 else None)
    ax2.axhline(1, color="k", lw=0.8)
    ax2.set_xticks(range(len(LANGUAGES)))
    ax2.set_xticklabels(LANGUAGES)
    ax2.set_ylabel("Slowest / fastest repetition")
    ax2.set_title("(b) Spread of every configuration (n = 64–2048)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_pinned_vs_unpinned.png", dpi=170)


def main():
    config = load_config()
    raw = pd.read_csv(RESULTS / "raw_measurements.csv")
    mem = pd.read_csv(RESULTS / "memory.csv")
    summary = summarise(raw, mem)
    # Execution configuration of the run: every implementation is single-threaded and pinned (see environment.json)
    with open(RESULTS / "environment.json", encoding="utf-8") as f:
        env = json.load(f)
    summary["threads"] = 1
    summary["cpu_affinity"] = " ".join(map(str, env.get("cpu_affinity", [])))
    fits = fit_exponents(summary)
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.6g")
    fits.to_csv(RESULTS / "fit.csv", index=False, float_format="%.6g")
    plot(summary, config)
    plot_dispersion(raw)

    pd.set_option("display.width", 200)
    cols = ["language", "n", "status", "warmups", "measured_reps", "median", "q1", "q3", "iqr_pct_of_median",
            "observed_ratio", "cubic_ratio", "ns_per_multiply_add", "time_relative_to_C", "peak_working_set_mb",
            "checksum_consistent"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print()
    print(fits.to_string(index=False))
    both = compare_preliminary(summary)
    if both is not None:
        print()
        print(both[["language", "n", "median", "pinned_max_over_min", "unpinned_median", "unpinned_max_over_min"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
