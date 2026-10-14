"""Basic triple-loop matrix multiplication in Python (Assignment 1).

Usage:
  python matmul.py bench <n> <warmups> <reps> <data_dir>
      Loads A_<n>.bin and B_<n>.bin, runs <warmups> unreported multiplications and <reps>
      measured ones. Prints one line per multiplication: "<warmup|rep> <seconds> <checksum_bits>".
  python matmul.py multiply <n> <A.bin> <B.bin> <C.bin>
      Multiplies the given matrices once and writes C (used by the correctness tests).

Matrices are stored on disk as raw little-endian float64 values in row-major order.
In memory they are lists of lists of Python floats (one list per row).
"""
import struct
import sys
import time
from array import array


def load(path, n):
    values = array("d")
    with open(path, "rb") as f:
        values.fromfile(f, n * n)
    return [values[i * n:(i + 1) * n].tolist() for i in range(n)]


def save(path, C):
    values = array("d")
    for row in C:
        values.extend(row)
    with open(path, "wb") as f:
        values.tofile(f)


def multiply(A, B, n):
    """Classic i-j-k triple loop. Allocation of C is part of the timed kernel."""
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def checksum_bits(C):
    """Sequential row-major sum of C, returned as the raw IEEE-754 bits of the double.

    An explicit loop is used because the built-in sum() uses compensated summation since
    Python 3.12, which would not match the plain summation done in Java and C.
    """
    total = 0.0
    for row in C:
        for x in row:
            total += x
    return struct.unpack("<q", struct.pack("<d", total))[0]


def main():
    mode = sys.argv[1]
    if mode == "bench":
        n, warmups, reps, data_dir = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        A = load(f"{data_dir}/A_{n}.bin", n)
        B = load(f"{data_dir}/B_{n}.bin", n)
        for r in range(warmups + reps):
            start = time.perf_counter()
            C = multiply(A, B, n)
            elapsed = time.perf_counter() - start
            label = "warmup" if r < warmups else "rep"
            print(f"{label} {elapsed:.9f} {checksum_bits(C)}", flush=True)
    elif mode == "multiply":
        n = int(sys.argv[2])
        save(sys.argv[5], multiply(load(sys.argv[3], n), load(sys.argv[4], n), n))
    else:
        sys.exit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
