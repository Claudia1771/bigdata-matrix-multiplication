/*
 * Basic triple-loop matrix multiplication in C (Assignment 1).
 *
 * Build: gcc -O2 -std=c11 -o matmul matmul.c
 * Usage:
 *   matmul bench <n> <warmups> <reps> <data_dir>
 *       Loads A_<n>.bin and B_<n>.bin, runs <warmups> unreported multiplications and <reps>
 *       measured ones. Prints one line per multiplication: "<warmup|rep> <seconds> <checksum_bits>".
 *   matmul multiply <n> <A.bin> <B.bin> <C.bin>
 *       Multiplies the given matrices once and writes C (used by the correctness tests).
 *
 * Matrices are stored on disk as raw little-endian float64 values in row-major order.
 * In memory they are contiguous one-dimensional arrays in row-major order: M[i][j] = M[i * n + j].
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double *load(const char *path, int n) {
    double *M = malloc(sizeof(double) * n * n);
    FILE *f = fopen(path, "rb");
    if (!M || !f || fread(M, sizeof(double), (size_t)n * n, f) != (size_t)n * n) {
        fprintf(stderr, "Cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    return M;
}

static void save(const char *path, const double *C, int n) {
    FILE *f = fopen(path, "wb");
    fwrite(C, sizeof(double), (size_t)n * n, f);
    fclose(f);
}

/* Classic i-j-k triple loop. Allocation of C is part of the timed kernel. */
static double *multiply(const double *A, const double *B, int n) {
    double *C = malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
    return C;
}

/* Sequential row-major sum of C, returned as the raw IEEE-754 bits of the double. */
static int64_t checksum_bits(const double *C, int n) {
    double total = 0.0;
    for (long i = 0; i < (long)n * n; i++)
        total += C[i];
    int64_t bits;
    memcpy(&bits, &total, sizeof bits);
    return bits;
}

int main(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "bench") == 0) {
        int n = atoi(argv[2]), warmups = atoi(argv[3]), reps = atoi(argv[4]);
        char path[1024];
        snprintf(path, sizeof path, "%s/A_%d.bin", argv[5], n);
        double *A = load(path, n);
        snprintf(path, sizeof path, "%s/B_%d.bin", argv[5], n);
        double *B = load(path, n);
        for (int r = 0; r < warmups + reps; r++) {
            double start = now();
            double *C = multiply(A, B, n);
            double elapsed = now() - start;
            printf("%s %.9f %" PRId64 "\n", r < warmups ? "warmup" : "rep", elapsed, checksum_bits(C, n));
            fflush(stdout);
            free(C);
        }
        free(A);
        free(B);
    } else if (argc >= 6 && strcmp(argv[1], "multiply") == 0) {
        int n = atoi(argv[2]);
        double *A = load(argv[3], n), *B = load(argv[4], n);
        double *C = multiply(A, B, n);
        save(argv[5], C, n);
        free(A); free(B); free(C);
    } else {
        fprintf(stderr, "Usage: %s bench <n> <warmups> <reps> <data_dir> | multiply <n> <A> <B> <C>\n", argv[0]);
        return 1;
    }
    return 0;
}
