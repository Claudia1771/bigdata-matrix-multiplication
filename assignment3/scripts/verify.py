"""Correctness tests for the sequential and parallel methods, run before benchmarking.

For several sizes (including sizes that are not multiples of the block, and fewer rows of blocks than threads)
and thread counts (including more threads than logical processors), both methods are compared with:
  - known products (1x1, 2x2, A x I = A, A x 0 = 0): exact comparison;
  - numpy.matmul (trusted reference) with the tolerances of config.json;
  - the sequential method, bit by bit (the parallel method must give exactly the same bits).
Both schedules (dynamic and static) are tested. Output: results/correctness.csv
"""
import csv
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from common import RESULTS, build, load_config


def multiply(exe, method, threads, schedule, A, B, tmp, block=64):
    n = A.shape[0]
    a, b, c = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
    A.astype("<f8").tofile(a)
    B.astype("<f8").tofile(b)
    subprocess.run([exe, "multiply", method, str(n), str(threads), str(a), str(b), str(c), str(block), schedule],
                   check=True, env={**os.environ, "OMP_DYNAMIC": "false"})
    return np.fromfile(c, dtype="<f8").reshape(n, n)


def main():
    config = load_config()
    exe = build(config)
    rtol, atol = config["tolerance"]["rtol"], config["tolerance"]["atol"]
    rng = np.random.default_rng(config["seed"])
    A8 = rng.random((8, 8))
    cases = [("known_1x1", np.array([[3.0]]), np.array([[4.0]]), np.array([[12.0]]), True),
             ("known_2x2", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]),
              np.array([[19.0, 22.0], [43.0, 50.0]]), True),
             ("identity_8x8", A8, np.eye(8), A8, True),
             ("zero_8x8", A8, np.zeros((8, 8)), np.zeros((8, 8)), True)]
    for n in config["correctness"]["sizes"]:
        A, B = rng.random((n, n)), rng.random((n, n))
        cases.append((f"random_{n}x{n}", A, B, A @ B, False))

    rows, all_ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case, A, B, expected, exact in cases:
            seq = multiply(exe, "seq", 1, "dynamic", A, B, tmp)
            runs = [("seq", 1, "-", seq)]
            for threads in config["correctness"]["threads"]:
                for schedule in ("dynamic", "static"):
                    runs.append(("par", threads, schedule, multiply(exe, "par", threads, schedule, A, B, tmp)))
            for method, threads, schedule, C in runs:
                ok = np.array_equal(C, expected) if exact else np.allclose(C, expected, rtol=rtol, atol=atol)
                same = bool(np.array_equal(C.view(np.int64), seq.view(np.int64)))
                ok = bool(ok and same)
                all_ok &= ok
                rows.append({"case": case, "method": method, "threads": threads, "schedule": schedule,
                             "max_abs_error": f"{np.abs(C - expected).max():.3e}",
                             "bitwise_equal_to_seq": same, "passed": ok})
            print(f"{case:15s} {len(runs)} runs  {'PASS' if all(r['passed'] for r in rows[-len(runs):]) else 'FAIL'}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "correctness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{sum(r['passed'] for r in rows)} of {len(rows)} tests passed")
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
