# Assignment 3 – Parallel Multiplication

Research question: *How does the number of threads affect the time, speedup and efficiency of a shared-memory
parallel version of the blocked matrix multiplication of Assignment 2, on a hybrid processor with performance and
efficient cores?*

The report is [`report/Assignment03_AlvarezGonzalez_Report.pdf`](report/Assignment03_AlvarezGonzalez_Report.pdf).
The optional vectorisation extension was not undertaken.

## Reuse of earlier assignments

- The sequential baseline `seq` is the blocked (tiled, 64×64) method of Assignment 2, which was the fastest dense
  method there; the parallel method `par` uses exactly the same kernel.
- The inputs are the dense matrices of Assignment 2 (identical to Assignment 1: `numpy.random.default_rng(42 + n)`).
- The measurement protocol and harness are those of Assignment 2.

## Contents

| Path | Description |
|---|---|
| `src/parallel.c` | `seq` and `par` (OpenMP): both call the same vectorised `block_row` kernel; `par` distributes the block rows of C among the threads (dynamic or static schedule) |
| `config.json` | Sizes, thread counts, affinity plan, schedule comparison, repetitions, budgets, flags |
| `scripts/verify.py` | Correctness tests for every thread count and schedule → `results/correctness.csv` |
| `scripts/run_experiment.py` | Benchmark → `results/raw_measurements.csv`, `results/runs.csv`, `results/environment.json` |
| `scripts/analyze.py` | Speedup, efficiency, tables and figures → `results/summary.csv`, `results/speedup.csv`, `results/schedule.csv`, `results/fig*.png` |
| `report/` | Report source (LaTeX) and final PDF |

## Parallel design

- Work unit: one row of 64×64 blocks of C. Each thread writes only the rows of C it owns, and every `C[i][j]`
  still adds its products in increasing `k` order, so no synchronisation is needed and the result is bit-identical
  to the sequential method for any thread count (checked by `verify.py` and by a checksum in every run).
- Threads are set explicitly with `omp_set_num_threads`, `OMP_DYNAMIC=false`; the team size actually obtained is
  printed in every measurement. No numerical library is used, so there are no hidden library threads.

## Affinity plan (Intel Core Ultra 7 155H: 6 P-cores × 2 threads, 8 E-cores, 2 LP E-cores = 22 logical)

| Threads p | Logical processors allowed |
|---|---|
| `seq`, 1 | 12 (one performance core) |
| 2–6 | one hardware thread of each performance core: 12, 0, 10, 14, 16, 18 |
| 7–12 | plus their hyper-thread siblings: 13, 1, 11, 15, 17, 19 |
| 16, 22 | all 22 logical processors (efficient and low-power cores included) |
| 44 | all 22 logical processors, 2× oversubscription |

The sibling pairs were identified by running two CPU-bound processes pinned to two logical processors: each one
becomes twice as slow only when both share a physical core.

## Requirements

- GCC 16.2.0 with OpenMP (`-fopenmp`, MinGW-w64 on Windows); Python 3.13 with `requirements.txt`
- Windows 11 (peak memory through the Win32 API); the reported run used AC power and the *Best performance* mode

## How to reproduce

From the `assignment3` folder (the inputs are created by `assignment2/scripts/generate_inputs.py`):

```bash
cd scripts
pip install -r ../requirements.txt
python verify.py
python run_experiment.py       # about 15 minutes; refuses to start on battery
python analyze.py
```

Build command used by the scripts: `gcc -O2 -std=c11 -fopenmp -o build/parallel.exe src/parallel.c`.
