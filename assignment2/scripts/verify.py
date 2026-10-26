"""Correctness tests for every dense method and for the CSR multiplication, run before benchmarking.

Dense methods (naive, ikj, transpose, tiled with several block sizes):
  - known 1x1 and 2x2 products, A x I = A and A x 0 = 0 (exact comparison);
  - random matrices of several sizes, including sizes that are not multiples of the block size,
    compared with numpy.matmul (trusted reference) using the tolerances in config.json;
  - bitwise comparison of every method with the naive baseline.
CSR multiplication (own code):
  - identity and a matrix with empty rows (exact comparison);
  - random uniform and banded matrices, compared with scipy.sparse (trusted reference): same sparsity
    pattern, values within the tolerances, and bitwise comparison with the dense ikj product.
Output: results/correctness.csv
"""
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from common import RESULTS, build, load_config, read_csr, write_csr
from generate_inputs import random_sparse

METHODS = [("naive", 64), ("ikj", 64), ("transpose", 64), ("tiled", 16), ("tiled", 64), ("tiled", 256)]


def dense_multiply(exe, method, block, A, B, tmp):
    n = A.shape[0]
    a, b, c = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
    A.astype("<f8").tofile(a)
    B.astype("<f8").tofile(b)
    subprocess.run([exe, "multiply", method, str(n), str(a), str(b), str(c), str(block)], check=True)
    return np.fromfile(c, dtype="<f8").reshape(n, n)


def sparse_multiply(exe, A, B, tmp):
    paths = [tmp / f"{x}.csr" for x in "ABC"]
    for M, p in zip((A, B), paths):
        M = M.tocsr()
        M.sort_indices()
        write_csr(p, M.shape[0], M.indptr, M.indices, M.data)
    subprocess.run([exe, "multiply", *map(str, paths)], check=True)
    n, row_ptr, col, val = read_csr(paths[2])
    return sps.csr_matrix((val, col, row_ptr), shape=(n, n))


def main():
    config = load_config()
    programs = build(config)
    rtol, atol = config["tolerance"]["rtol"], config["tolerance"]["atol"]
    rng = np.random.default_rng(config["seed"])
    rows, all_ok = [], True

    def record(part, case, method, ok, max_abs, bitwise):
        nonlocal all_ok
        all_ok &= bool(ok)
        rows.append({"part": part, "case": case, "method": method, "max_abs_error": f"{max_abs:.3e}",
                     "bitwise_equal_to_baseline": bitwise, "passed": bool(ok)})
        print(f"{part:6s} {case:28s} {method:12s} max_abs={max_abs:.2e} bitwise={bitwise}  {'PASS' if ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ---- Part A: dense methods -------------------------------------------------------------
        A8 = rng.random((8, 8))
        cases = [("known_1x1", np.array([[3.0]]), np.array([[4.0]]), np.array([[12.0]]), True),
                 ("known_2x2", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]),
                  np.array([[19.0, 22.0], [43.0, 50.0]]), True),
                 ("identity_8x8", A8, np.eye(8), A8, True),
                 ("zero_8x8", A8, np.zeros((8, 8)), np.zeros((8, 8)), True)]
        for n in config["correctness"]["dense_sizes"]:
            A, B = rng.random((n, n)), rng.random((n, n))
            cases.append((f"random_{n}x{n}", A, B, A @ B, False))
        for case, A, B, expected, exact in cases:
            baseline = None
            for method, block in METHODS:
                C = dense_multiply(programs["dense"], method, block, A, B, tmp)
                baseline = C if baseline is None else baseline
                ok = np.array_equal(C, expected) if exact else np.allclose(C, expected, rtol=rtol, atol=atol)
                label = f"{method}" + (f"(b={block})" if method == "tiled" else "")
                record("dense", case, label, ok, np.abs(C - expected).max(),
                       bool(np.array_equal(C.view(np.int64), baseline.view(np.int64))))

        # ---- Part B: CSR multiplication ----------------------------------------------------------
        n = 50
        I = sps.identity(n, format="csr")
        B = sps.random(n, n, density=0.2, random_state=1, format="csr")
        C = sparse_multiply(programs["sparse"], I, B, tmp)
        record("sparse", "identity_x_B_50", "spgemm", (C != B).nnz == 0, abs(C - B).max() if B.nnz else 0, True)
        A = sps.random(n, n, density=0.1, random_state=2, format="lil")
        A[5, :] = 0
        A[17, :] = 0
        A = A.tocsr()
        A.eliminate_zeros()
        C = sparse_multiply(programs["sparse"], A, B, tmp)
        ref = A @ B
        ok = C.shape == ref.shape and np.allclose(C.toarray(), ref.toarray(), rtol=rtol, atol=atol) \
            and C[5].nnz == 0 and C[17].nnz == 0
        record("sparse", "empty_rows_50", "spgemm", ok, abs(C - ref).max(), True)

        for n, d, structure in config["correctness"]["sparse_cases"]:
            mats = []
            for matrix_id in (0, 1):
                row_ptr, col, val, _ = random_sparse(n, structure, d, matrix_id, config["seed"] + 1000)
                mats.append(sps.csr_matrix((val, col, row_ptr), shape=(n, n)))
            A, B = mats
            C = sparse_multiply(programs["sparse"], A, B, tmp)
            ref = (A @ B).tocsr()
            ref.sort_indices()
            same_pattern = C.nnz == ref.nnz and np.array_equal(C.indptr, ref.indptr) and np.array_equal(C.indices, ref.indices)
            close = np.allclose(C.toarray(), ref.toarray(), rtol=rtol, atol=atol)
            dense_ikj = dense_multiply(programs["dense"], "ikj", 64, A.toarray(), B.toarray(), tmp)
            bitwise = bool(np.array_equal(C.toarray().view(np.int64), dense_ikj.view(np.int64)))
            record("sparse", f"{structure}_n{n}_d{d}_nnzC{C.nnz}", "spgemm", same_pattern and close and bitwise,
                   abs(C - ref).max(), bitwise)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "correctness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
