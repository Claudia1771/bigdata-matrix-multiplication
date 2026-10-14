"""Generate the shared input matrices used by all three implementations.

Each matrix is written as raw little-endian float64 values in row-major order:
  data/A_<n>.bin, data/B_<n>.bin
A and B for size n are drawn from numpy.random.default_rng(seed + n), so every size is reproducible
independently. A SHA-256 hash of every file is stored in data/manifest.csv.
"""
import csv
import hashlib

import numpy as np

from common import DATA, load_config


def main():
    config = load_config()
    DATA.mkdir(exist_ok=True)
    rows = []
    for n in sorted(set(config["sizes"])):
        rng = np.random.default_rng(config["seed"] + n)
        for name in ("A", "B"):
            M = rng.random((n, n), dtype=np.float64)
            path = DATA / f"{name}_{n}.bin"
            M.astype("<f8").tofile(path)
            rows.append({"file": path.name, "n": n, "seed": config["seed"] + n,
                         "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"{path.name}: {n}x{n}, {path.stat().st_size / 2**20:.1f} MiB")
    with open(DATA / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
