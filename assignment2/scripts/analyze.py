"""Summarise the measurements of Assignment 2 and produce the tables and figures of the report.

Inputs:  results/raw_measurements.csv, results/runs.csv, results/selection.json, results/inputs_manifest.csv
         and the CSR input files (to count the scalar multiply-adds of every sparse product).
Outputs: results/summary.csv                 one row per configuration (median, quartiles, IQR, memory, ...)
         results/sparse_comparison.csv       spgemm vs dense vs conversion for every matched sparse input
         results/break_even.csv              density at which spgemm and dense take the same time
         results/fig1_block_tuning.png       tiled method: median time vs block size
         results/fig2_dense_time.png         Part A: median time vs n (log-log, IQR bars)
         results/fig3_dense_per_op.png       Part A: time per multiply-add vs n
         results/fig4_dense_speedup.png      Part A: speedup over the naive baseline
         results/fig5_sparse_density.png     Part B: time vs density at fixed n (both structures)
         results/fig6_fill_in.png            Part B: output density vs input density
         results/fig7_sparse_scaling.png     Part B: time and memory vs n at fixed density
         results/fig8_core_frequency.png     Part A: effective clock of the pinned core per configuration
Quartiles use numpy.percentile with linear interpolation.
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, load_config, read_csr, sparse_paths

DENSE_COLORS = {"naive": "#9a9a9a", "ikj": "#3b6fb6", "transpose": "#d9822b", "tiled": "#2a9d5c"}
DENSE_MARKERS = {"naive": "o", "ikj": "s", "transpose": "D", "tiled": "^"}
STRUCT_STYLE = {"uniform": ("#3b6fb6", "o"), "banded": ("#c0392b", "s")}


def summarise(raw, runs):
    reps = raw[raw["kind"] == "rep"]
    keys = ["experiment", "method", "n", "block", "structure", "density_pct"]
    stats = reps.groupby(keys, dropna=False).agg(
        median=("time_s", "median"), q1=("time_s", lambda x: np.percentile(x, 25)),
        q3=("time_s", lambda x: np.percentile(x, 75)), min=("time_s", "min"), max=("time_s", "max"),
        measured_reps=("time_s", "count"), nnz_out=("nnz_out", "first"),
        checksums=("checksum_bits", "nunique")).reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["iqr_pct_of_median"] = 100 * stats["iqr"] / stats["median"]
    summary = runs.merge(stats, on=keys, how="left")
    summary["threads"] = 1
    return summary


def sparse_work(config, n, structure, density):
    """Number of scalar multiply-adds of the CSR product: sum over the nonzeros A[i,k] of nnz(row k of B)."""
    a_path, b_path = sparse_paths(config, n, structure, density)
    _, a_ptr, a_col, _ = read_csr(a_path)
    _, b_ptr, _, _ = read_csr(b_path)
    return int(np.diff(b_ptr)[a_col].sum()), len(a_col)


def log_interp_crossing(x, y1, y2):
    """x where y1 == y2, interpolating log(y1/y2) linearly in log(x). None if there is no sign change."""
    r = np.log(np.asarray(y1) / np.asarray(y2))
    lx = np.log(np.asarray(x))
    for i in range(len(r) - 1):
        if r[i] == 0:
            return x[i]
        if r[i] * r[i + 1] < 0:
            return float(np.exp(lx[i] - r[i] * (lx[i + 1] - lx[i]) / (r[i + 1] - r[i])))
    return None


def main():
    config = load_config()
    raw = pd.read_csv(RESULTS / "raw_measurements.csv")
    runs = pd.read_csv(RESULTS / "runs.csv")
    selection = json.loads((RESULTS / "selection.json").read_text())
    for df in (raw, runs):
        df["block"] = pd.to_numeric(df["block"], errors="coerce")
    summary = summarise(raw, runs)

    # ---------------- Part A -------------------------------------------------------------------
    dense = summary[summary["experiment"] == "dense"].copy()
    dense["ns_per_multiply_add"] = 1e9 * dense["median"] / dense["n"].astype(float) ** 3
    base = dense[dense["method"] == "naive"].set_index("n")["median"]
    dense["speedup_vs_naive"] = dense["n"].map(base) / dense["median"]

    tuning = summary[summary["experiment"] == "block_tuning"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n, g in tuning.groupby("n"):
        g = g.sort_values("block")
        ax.errorbar(g["block"], g["median"], yerr=[g["median"] - g["q1"], g["q3"] - g["median"]], marker="o",
                    capsize=3, label=f"n = {n}")
    ax.axvline(selection["block"], color="grey", ls=":", label=f"selected block = {selection['block']}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(config["dense"]["block_tuning"]["blocks"])
    ax.set_xticklabels(config["dense"]["block_tuning"]["blocks"])
    ax.set_xlabel("Block size (elements per side)")
    ax.set_ylabel("Median time per multiplication (s)")
    ax.set_title("Tiled method: effect of the block size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_block_tuning.png", dpi=170)

    sizes = sorted(dense["n"].unique())
    pow2 = [n for n in sizes if n & (n - 1) == 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in config["dense"]["methods"]:
        g = dense[(dense["method"] == m) & (dense["status"] == "ok")].sort_values("n")
        ax.errorbar(g["n"], g["median"], yerr=[g["median"] - g["q1"], g["q3"] - g["median"]], marker=DENSE_MARKERS[m],
                    color=DENSE_COLORS[m], capsize=3, label=m)
        failed = dense[(dense["method"] == m) & (dense["status"] == "timeout")]
        if len(failed):
            ax.scatter(failed["n"], [config["time_limit_s"]] * len(failed), marker="x", s=80, color=DENSE_COLORS[m],
                       zorder=5, label=f"{m}: timeout (> {config['time_limit_s']} s)")
    ax.axhline(config["time_limit_s"], color="grey", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(pow2)
    ax.set_xticklabels(pow2)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Median time per multiplication (s)")
    ax.set_title("Part A: dense methods (bars: IQR; point left of 1024 is n = 1000)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_dense_time.png", dpi=170)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for m in config["dense"]["methods"]:
        g = dense[(dense["method"] == m) & (dense["status"] == "ok")].sort_values("n")
        ax.plot(g["n"], g["ns_per_multiply_add"], marker=DENSE_MARKERS[m], color=DENSE_COLORS[m], label=m)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(pow2)
    ax.set_xticklabels(pow2)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Median time / n³ (ns per multiply-add)")
    ax.set_title("Part A: cost per multiply-add (constant = exactly cubic growth)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_dense_per_op.png", dpi=170)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    opt = [m for m in config["dense"]["methods"] if m != "naive"]
    ok = dense[(dense["status"] == "ok") & dense["speedup_vs_naive"].notna()]
    width = 0.8 / len(opt)
    common_sizes = sorted(ok[ok["method"] == "naive"]["n"].unique())
    for j, m in enumerate(opt):
        g = ok[ok["method"] == m].set_index("n").reindex(common_sizes)
        x = np.arange(len(common_sizes)) + (j - (len(opt) - 1) / 2) * width
        ax.bar(x, g["speedup_vs_naive"], width=width, color=DENSE_COLORS[m], label=m)
        for xi, v in zip(x, g["speedup_vs_naive"]):
            ax.text(xi, v * 1.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(common_sizes)))
    ax.set_xticklabels(common_sizes)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Speedup over naive (median ratio)")
    ax.set_title("Part A: speedup of the optimised methods over the baseline")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_dense_speedup.png", dpi=170)

    # Effective clock of the pinned core during each dense configuration (recorded by the harness)
    if "core_freq_mhz" in dense:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for m in config["dense"]["methods"]:
            g = dense[(dense["method"] == m) & dense["core_freq_mhz"].notna()].sort_values("n")
            ax.plot(g["n"], g["core_freq_mhz"] / 1000, marker=DENSE_MARKERS[m], color=DENSE_COLORS[m], label=m)
        ax.set_xscale("log", base=2)
        ax.set_xticks(pow2)
        ax.set_xticklabels(pow2)
        ax.set_xlabel("Matrix size n (n × n)")
        ax.set_ylabel("Mean effective clock of the core (GHz)")
        ax.set_title("Part A: effective clock of the pinned core while each configuration ran")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig8_core_frequency.png", dpi=170)

    # ---------------- Part B -------------------------------------------------------------------
    sparse = summary[summary["experiment"].isin(["sparse_density", "sparse_scaling"])].copy()
    keys = ["experiment", "n", "structure", "density_pct"]
    wide = sparse.pivot_table(index=keys, columns="method", values=["median", "q1", "q3", "nnz_out",
                                                                     "peak_working_set_mb"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    work, nnz_in = [], []
    for _, r in wide.iterrows():
        w, nz = sparse_work(config, int(r["n"]), r["structure"], r["density_pct"])
        work.append(w)
        nnz_in.append(nz)
    wide["nnz_A"] = nnz_in
    wide["multiply_adds_spgemm"] = work
    wide["nnz_C"] = wide["nnz_out_spgemm"]
    wide["density_C_pct"] = 100 * wide["nnz_C"] / wide["n"].astype(float) ** 2
    wide["speedup_spgemm_vs_dense"] = wide["median_dense"] / wide["median_spgemm"]
    wide["speedup_incl_conversion"] = wide["median_dense"] / (wide["median_spgemm"] + wide["median_convert"])
    wide["ns_per_multiply_add_spgemm"] = 1e9 * wide["median_spgemm"] / wide["multiply_adds_spgemm"]
    wide["ns_per_multiply_add_dense"] = 1e9 * wide["median_dense"] / wide["n"].astype(float) ** 3
    wide.to_csv(RESULTS / "sparse_comparison.csv", index=False, float_format="%.6g")

    dens = wide[wide["experiment"] == "sparse_density"].sort_values("density_pct")
    rows = []
    for s, g in dens.groupby("structure"):
        rows.append({"structure": s, "n": config["sparse"]["n"],
                     "break_even_density_pct": log_interp_crossing(g["density_pct"].values, g["median_spgemm"].values,
                                                                   g["median_dense"].values),
                     "break_even_incl_conversion_pct": log_interp_crossing(
                         g["density_pct"].values, (g["median_spgemm"] + g["median_convert"]).values,
                         g["median_dense"].values)})
    pd.DataFrame(rows).to_csv(RESULTS / "break_even.csv", index=False, float_format="%.6g")

    fig, ax = plt.subplots(figsize=(8, 5))
    for s, g in dens.groupby("structure"):
        color, marker = STRUCT_STYLE[s]
        ax.errorbar(g["density_pct"], g["median_spgemm"], yerr=[g["median_spgemm"] - g["q1_spgemm"],
                    g["q3_spgemm"] - g["median_spgemm"]], marker=marker, color=color, capsize=3,
                    label=f"CSR spgemm, {s}")
        ax.plot(g["density_pct"], g["median_convert"], marker=marker, color=color, ls=":", alpha=0.8,
                label=f"dense→CSR conversion, {s}")
    g = dens[dens["structure"] == "uniform"]
    ax.errorbar(g["density_pct"], g["median_dense"], yerr=[g["median_dense"] - g["q1_dense"],
                g["q3_dense"] - g["median_dense"]], marker="^", color="#2a9d5c", capsize=3,
                label=f"dense tiled (same matrices, b = {selection['block']})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input density (% of nonzero entries); sparsity = 100 − density")
    ax.set_ylabel("Median time (s)")
    ax.set_title(f"Part B: sparse vs dense, n = {config['sparse']['n']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_sparse_density.png", dpi=170)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for s, g in dens.groupby("structure"):
        color, marker = STRUCT_STYLE[s]
        ax.plot(g["density_pct"], g["density_C_pct"], marker=marker, color=color, label=f"measured, {s}")
    d = np.logspace(np.log10(dens["density_pct"].min()), np.log10(dens["density_pct"].max()), 100)
    n = config["sparse"]["n"]
    ax.plot(d, 100 * (1 - (1 - (d / 100) ** 2) ** n), "k--", lw=1, label="uniform model 1 − (1 − d²)ⁿ")
    ax.plot(d, d, color="grey", lw=0.8, ls=":", label="output density = input density")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input density (%)")
    ax.set_ylabel("Density of C = AB (%)")
    ax.set_title(f"Part B: fill-in of the product, n = {n}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig6_fill_in.png", dpi=170)

    scal = summary[summary["experiment"] == "sparse_scaling"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = {"spgemm": ("#3b6fb6", "o", "CSR spgemm"), "dense": ("#2a9d5c", "^", "dense tiled"),
              "convert": ("#d9822b", "s", "dense→CSR conversion")}
    for mode, (color, marker, label) in styles.items():
        g = scal[(scal["method"] == mode) & (scal["status"] == "ok")].sort_values("n")
        ax1.errorbar(g["n"], g["median"], yerr=[g["median"] - g["q1"], g["q3"] - g["median"]], marker=marker,
                     color=color, capsize=3, label=label)
        ax2.plot(g["n"], g["peak_working_set_mb"].astype(float), marker=marker, color=color, label=label)
        skipped = scal[(scal["method"] == mode) & (scal["status"] != "ok")]
        if len(skipped):
            ax2.scatter(skipped["n"], skipped["predicted_memory_mb"], marker="x", s=70, color=color,
                        label=f"{label}: not run (predicted)")
    ax2.axhline(config["memory_budget_mb"], color="red", lw=0.8, ls="--", label="memory budget")
    sc = config["sparse"]["scaling"]
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sc["sizes"])
        ax.set_xticklabels(sc["sizes"], fontsize=8)
        ax.set_xlabel("Matrix size n (n × n)")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("Median time (s)")
    ax1.set_title(f"(a) Time, {sc['structure']} density {sc['density_pct']}%")
    ax2.set_ylabel("Peak working set (MiB)")
    ax2.set_title("(b) Peak memory of the process")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig7_sparse_scaling.png", dpi=170)

    summary = summary.merge(dense[["experiment", "method", "n", "ns_per_multiply_add", "speedup_vs_naive"]],
                            on=["experiment", "method", "n"], how="left")
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.6g")

    pd.set_option("display.width", 220)
    print(tuning[["n", "block", "median", "iqr_pct_of_median"]].to_string(index=False))
    cols = ["method", "n", "status", "median", "iqr_pct_of_median", "ns_per_multiply_add", "speedup_vs_naive",
            "peak_working_set_mb"] + [c for c in ("core_freq_mhz", "sibling_cpu_pct") if c in dense]
    print(dense[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(wide[["experiment", "n", "structure", "density_pct", "nnz_A", "nnz_C", "density_C_pct", "median_spgemm",
                "median_dense", "median_convert", "speedup_spgemm_vs_dense", "speedup_incl_conversion",
                "ns_per_multiply_add_spgemm", "peak_working_set_mb_spgemm", "peak_working_set_mb_dense"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(pd.read_csv(RESULTS / "break_even.csv").to_string(index=False))
    bad = summary[(summary["checksums"] > 1)]
    print("\nConfigurations with more than one checksum:", len(bad))

    # Execution conditions recorded by the harness for every configuration
    ran = summary[summary["status"] == "ok"]
    if "on_ac_power" in ran:
        print("Configurations not on AC power:", int((~ran["on_ac_power"].astype(bool)).sum()))
        print(f"System CPU utilisation during the runs: median {ran['system_cpu_pct'].median():.1f}%, "
              f"max {ran['system_cpu_pct'].max():.1f}%")
        print(f"Utilisation of the sibling logical processor: median {ran['sibling_cpu_pct'].median():.1f}%, "
              f"max {ran['sibling_cpu_pct'].max():.1f}%")
        busy = ran[ran["sibling_cpu_pct"] > 10]
        print(busy[["experiment", "method", "n", "structure", "density_pct", "sibling_cpu_pct", "iqr_pct_of_median"]]
              .to_string(index=False) if len(busy) else "No configuration with sibling utilisation above 10%")


if __name__ == "__main__":
    main()
