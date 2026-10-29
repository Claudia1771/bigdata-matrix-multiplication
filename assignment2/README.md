# Assignment 2 – Optimized Approaches and Sparse Matrices

Research questions:

- **Part A:** *How much do loop reordering, transposition and cache blocking reduce the time of the basic dense
  matrix multiplication of Assignment 1, and how does the gain change with the matrix size?*
- **Part B:** *Under which tested densities and structures does a CSR sparse multiplication outperform a dense
  multiplication of the same matrices, once conversion costs and the density of the result are considered?*

The report is [`report/Assignment02_AlvarezGonzalez_Report.pdf`](report/Assignment02_AlvarezGonzalez_Report.pdf).

## Reuse of Assignment 1

- The baseline `naive` in `src/dense.c` is the validated C implementation of Assignment 1 (same code).
- The dense inputs are generated exactly as in Assignment 1 (`numpy.random.default_rng(42 + n)`); the SHA-256
  hashes of the shared sizes in `results/inputs_manifest.csv` are identical to `assignment1/data/manifest.csv`.
- The measurement protocol (pinned performance core, warm-ups, median and IQR, budgets, peak working set) and the
  harness are those of Assignment 1.

## Contents

| Path | Description |
|---|---|
| `src/dense.c` | Dense methods: `naive` (baseline), `ikj` (loop reordering), `transpose`, `tiled` (cache blocking) |
| `src/sparse.c` | Own CSR implementation: Gustavson CSR × CSR product, dense ↔ CSR conversion, dense comparison |
| `config.json` | Every parameter: seeds, sizes, densities, structures, repetitions, budgets, flags, affinity |
| `scripts/generate_inputs.py` | Dense and sparse inputs (written outside the repository, see `data_dir`) + `results/inputs_manifest.csv` |
| `scripts/verify.py` | Correctness tests against NumPy and SciPy → `results/correctness.csv` |
| `scripts/run_experiment.py` | The four experiments → `results/raw_measurements.csv`, `results/runs.csv`, `results/selection.json` |
| `scripts/analyze.py` | Summaries and figures → `results/summary.csv`, `results/sparse_comparison.csv`, `results/break_even.csv`, `results/fig*.png` |
| `report/` | Report source (LaTeX) and final PDF |

## Requirements

Versions used for the reported run (full details in `results/environment.json`):

- Python 3.13 with the packages in `requirements.txt` (NumPy for inputs, SciPy only as correctness reference)
- GCC 16.2.0 (MinGW-w64 on Windows; `gcc` in `PATH`), flags `-O2 -std=c11`
- Windows 11 (the memory measurement uses the Win32 `GetProcessMemoryInfo` API)

## How to reproduce

From the `assignment2` folder:

```bash
cd scripts
pip install -r ../requirements.txt
python generate_inputs.py      # about 0.5 GB of inputs in ~/matmul_data/assignment2
python verify.py               # builds both programs and runs the correctness tests
python run_experiment.py       # the four experiments (about 1-1.5 hours)
python analyze.py              # tables and figures
```

Single experiments can be repeated with `python run_experiment.py --experiments dense` (the stored results of
the other experiments are kept). The programs are built with:

```bash
gcc -O2 -std=c11 -o build/dense.exe src/dense.c
gcc -O2 -std=c11 -o build/sparse.exe src/sparse.c
```

Every benchmark process is started pinned to the logical processor in `cpu_affinity` (12, a performance core of
the Intel Core Ultra 7 155H used for the report) with high priority; on another machine choose one of its fastest
cores (see `assignment1/scripts/core_probe.py`) and set `sibling_cpu` to the other hardware thread of that core.

**Execution conditions.** The reported run used Windows' *Best performance* power mode on AC power. The harness
refuses to start on battery (`require_ac_power`), waits `cooldown_s` seconds before every configuration, and records
for each configuration in `results/runs.csv` whether it stayed on AC power, the utilisation of the system and of the
sibling logical processor, and the effective clock of the pinned core.

**Discarded runs.** `results/discarded_run_on_battery/` (laptop on battery) and
`results/interrupted_run_balanced_plan/` (*Balanced* power mode, stopped) are earlier runs kept as evidence; they are
not used in the report.

## Inputs

- **Dense:** raw little-endian `float64`, row-major, uniform values in [0, 1).
- **Sparse:** CSR files (`int64 n, int64 nnz, int64 row_ptr[n+1], int32 col[nnz], float64 val[nnz]`). Density is
  the percentage of nonzero entries (sparsity = 100 − density); `nnz = round(density/100 · n²)` positions are drawn
  without replacement either among all positions (`uniform`) or inside the narrowest diagonal band that can hold
  them (`banded`). Values are in (0, 1]. Seeds and SHA-256 hashes are in `results/inputs_manifest.csv`.
