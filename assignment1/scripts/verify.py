"""Correctness tests for the three implementations, run before benchmarking.

Test cases:
  - 1x1 and 2x2 matrices with known integer results (exact comparison)
  - A x I = A and A x 0 = 0 (exact comparison)
  - random matrices of several sizes, compared with numpy.matmul (trusted reference) using the
    tolerances in config.json
In addition, the results of the three languages are compared with each other bit by bit.
Output: results/correctness.csv
"""
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from common import LANGUAGES, RESULTS, build, load_config


def run_multiply(cmd, A, B, tmp):
    n = A.shape[0]
    a, b, c = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
    A.astype("<f8").tofile(a)
    B.astype("<f8").tofile(b)
    subprocess.run(cmd + ["multiply", str(n), str(a), str(b), str(c)], check=True)
    return np.fromfile(c, dtype="<f8").reshape(n, n)


def test_cases(config):
    rng = np.random.default_rng(config["seed"])
    yield "known_1x1", np.array([[3.0]]), np.array([[4.0]]), np.array([[12.0]]), True
    yield ("known_2x2", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]),
           np.array([[19.0, 22.0], [43.0, 50.0]]), True)
    A = rng.random((8, 8))
    yield "identity_8x8", A, np.eye(8), A, True
    yield "zero_8x8", A, np.zeros((8, 8)), np.zeros((8, 8)), True
    for n in config["correctness_sizes"]:
        A, B = rng.random((n, n)), rng.random((n, n))
        yield f"random_{n}x{n}", A, B, A @ B, False


def main():
    config = load_config()
    commands = build(config)
    rtol, atol = config["tolerance"]["rtol"], config["tolerance"]["atol"]
    RESULTS.mkdir(exist_ok=True)
    rows, all_passed = [], True

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case, A, B, expected, exact in test_cases(config):
            outputs = {lang: run_multiply(commands[lang], A, B, tmp) for lang in LANGUAGES}
            for lang, C in outputs.items():
                abs_err = np.abs(C - expected)
                rel_err = abs_err / np.maximum(np.abs(expected), np.finfo(float).tiny)
                passed = np.array_equal(C, expected) if exact else np.allclose(C, expected, rtol=rtol, atol=atol)
                all_passed &= bool(passed)
                rows.append({
                    "case": case, "n": A.shape[0], "language": lang,
                    "check": "exact" if exact else f"allclose(rtol={rtol}, atol={atol}) vs numpy.matmul",
                    "max_abs_error": f"{abs_err.max():.3e}", "max_rel_error": f"{rel_err.max():.3e}",
                    "bitwise_equal_to_C": bool(np.array_equal(C.view(np.int64), outputs["C"].view(np.int64))),
                    "passed": bool(passed),
                })
                print(f"{case:15s} {lang:6s} max_abs_err={abs_err.max():.2e} "
                      f"same_bits_as_C={rows[-1]['bitwise_equal_to_C']}  {'PASS' if passed else 'FAIL'}")

    with open(RESULTS / "correctness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nALL TESTS PASSED" if all_passed else "\nSOME TESTS FAILED")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
