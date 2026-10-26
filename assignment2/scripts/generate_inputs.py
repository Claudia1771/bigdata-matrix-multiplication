"""Generate the input matrices of Assignment 2 in config["data_dir"] and a manifest in results/inputs_manifest.csv.

Dense (Part A): A_<n>.bin and B_<n>.bin, raw little-endian float64, row-major. Generated exactly as in
Assignment 1 (numpy.random.default_rng(seed + n), A then B, uniform [0, 1)), so the sizes shared with
Assignment 1 reuse the same matrices (the SHA-256 hashes can be compared with assignment1/data/manifest.csv).

Sparse (Part B): CSR files (see src/sparse.c). For n, structure, density d (percentage of nonzero entries):
  nnz = round(d / 100 * n^2) positions are chosen without replacement
  - uniform: among all n^2 positions;
  - banded:  among the positions of the narrowest band |i - j| <= w that contains at least nnz positions.
  Values are 1 - uniform[0, 1), i.e. in (0, 1], so no stored value is zero.
  Seeds: numpy.random.default_rng([seed, n, structure_id, round(d * 1e4), matrix_id]) with matrix_id 0 for A, 1 for B.
Sparsity = 100 - density.
"""
import csv
import hashlib

import numpy as np

from common import RESULTS, STRUCTURES, dense_paths, load_config, sparse_paths, write_csr


def band_count(n, w):
    return n * (2 * w + 1) - w * (w + 1)


def band_positions(n, w):
    parts = []
    for off in range(-w, w + 1):
        i = np.arange(max(0, -off), min(n, n - off), dtype=np.int64)
        parts.append(i * n + i + off)
    return np.concatenate(parts)


def sample_without_replacement(rng, total, k):
    if k * 4 >= total:
        return rng.choice(total, size=k, replace=False)
    chosen = np.empty(0, dtype=np.int64)
    while len(chosen) < k:
        chosen = np.unique(np.concatenate([chosen, rng.integers(0, total, size=int((k - len(chosen)) * 1.2) + 16)]))
    return rng.choice(chosen, size=k, replace=False)


def random_sparse(n, structure, density_pct, matrix_id, seed):
    """Returns (row_ptr, col, val, band_half_width or None)."""
    rng = np.random.default_rng([seed, n, STRUCTURES[structure], round(density_pct * 1e4), matrix_id])
    nnz = max(1, round(density_pct / 100 * n * n))
    w = None
    if structure == "uniform":
        pos = sample_without_replacement(rng, n * n, nnz)
    else:
        w = 0
        while band_count(n, w) < nnz:
            w += 1
        candidates = band_positions(n, w)
        pos = candidates[rng.choice(len(candidates), size=nnz, replace=False)]
    pos = np.sort(pos)
    rows, cols = pos // n, pos % n
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])
    val = 1.0 - rng.random(nnz)
    return row_ptr, cols.astype(np.int32), val, w


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sparse_configurations(config):
    sp = config["sparse"]
    configs = {(sp["n"], s, d) for s in sp["structures"] for d in sp["densities_pct"]}
    sc = sp["scaling"]
    configs |= {(n, sc["structure"], sc["density_pct"]) for n in sc["sizes"]}
    return sorted(configs)


def main():
    config = load_config()
    RESULTS.mkdir(exist_ok=True)
    rows = []

    for n in config["dense"]["sizes"]:
        rng = np.random.default_rng(config["seed"] + n)
        for name, path in zip("AB", dense_paths(config, n)):
            if not path.exists():
                rng.random((n, n), dtype=np.float64).astype("<f8").tofile(path)
            else:
                rng.random((n, n), dtype=np.float64)  # keep the generator state identical
            rows.append({"part": "dense", "file": path.name, "n": n, "structure": "dense", "density_pct": 100,
                         "nnz": n * n, "band_half_width": "", "bytes": path.stat().st_size, "sha256": sha256(path)})
            print(f"{path.name}: dense {n}x{n}")

    for n, structure, d in sparse_configurations(config):
        for matrix_id, path in enumerate(sparse_paths(config, n, structure, d)):
            row_ptr, col, val, w = random_sparse(n, structure, d, matrix_id, config["seed"])
            write_csr(path, n, row_ptr, col, val)
            rows.append({"part": "sparse", "file": path.name, "n": n, "structure": structure, "density_pct": d,
                         "nnz": len(col), "band_half_width": "" if w is None else w,
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
            print(f"{path.name}: nnz={len(col)}" + ("" if w is None else f", band half-width {w}"))

    with open(RESULTS / "inputs_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
