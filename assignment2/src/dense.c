/*
 * Dense matrix multiplication methods (Assignment 2, Part A).
 *
 * Build: gcc -O2 -std=c11 -o dense dense.c
 * Usage:
 *   dense bench <method> <n> <warmups> <reps> <data_dir> [block]
 *       Loads A_<n>.bin and B_<n>.bin, runs <warmups> unreported and <reps> measured multiplications.
 *       Prints one line per multiplication: "<warmup|rep> <seconds> <checksum_bits>".
 *   dense multiply <method> <n> <A.bin> <B.bin> <C.bin> [block]
 *       One multiplication, used by the correctness tests.
 *
 * Methods (all perform exactly n^3 multiply-adds and add the products of each C[i][j] in increasing k order,
 * so they produce bit-identical results):
 *   naive      i-j-k triple loop: the validated baseline of Assignment 1 (same code)
 *   ikj        loop reordering: the innermost loop runs over j, so B and C are read row by row
 *   transpose  B is transposed first (inside the timed region), then i-j-k reads both operands row by row
 *   tiled      cache blocking: i-k-j order inside square blocks of side <block> (default 64)
 *
 * Matrices are raw little-endian float64 files in row-major order, stored in memory as contiguous arrays.
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

/* Baseline: identical to the C implementation of Assignment 1. */
static double *mm_naive(const double *A, const double *B, int n, int block) {
    (void)block;
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

static double *mm_ikj(const double *A, const double *B, int n, int block) {
    (void)block;
    double *C = calloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            const double *Bk = B + (size_t)k * n;
            double *Ci = C + (size_t)i * n;
            for (int j = 0; j < n; j++)
                Ci[j] += a * Bk[j];
        }
    return C;
}

static double *mm_transpose(const double *A, const double *B, int n, int block) {
    (void)block;
    double *BT = malloc(sizeof(double) * n * n);
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            BT[j * n + k] = B[k * n + j];
    double *C = malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++) {
        const double *Ai = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *BTj = BT + (size_t)j * n;
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += Ai[k] * BTj[k];
            C[i * n + j] = s;
        }
    }
    free(BT);
    return C;
}

static double *mm_tiled(const double *A, const double *B, int n, int block) {
    double *C = calloc((size_t)n * n, sizeof(double));
    for (int ii = 0; ii < n; ii += block)
        for (int kk = 0; kk < n; kk += block)
            for (int jj = 0; jj < n; jj += block) {
                int imax = ii + block < n ? ii + block : n;
                int kmax = kk + block < n ? kk + block : n;
                int jmax = jj + block < n ? jj + block : n;
                for (int i = ii; i < imax; i++) {
                    double *Ci = C + (size_t)i * n;
                    for (int k = kk; k < kmax; k++) {
                        double a = A[i * n + k];
                        const double *Bk = B + (size_t)k * n;
                        for (int j = jj; j < jmax; j++)
                            Ci[j] += a * Bk[j];
                    }
                }
            }
    return C;
}

typedef double *(*method_fn)(const double *, const double *, int, int);

static method_fn find_method(const char *name) {
    if (strcmp(name, "naive") == 0) return mm_naive;
    if (strcmp(name, "ikj") == 0) return mm_ikj;
    if (strcmp(name, "transpose") == 0) return mm_transpose;
    if (strcmp(name, "tiled") == 0) return mm_tiled;
    fprintf(stderr, "Unknown method: %s\n", name);
    exit(1);
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
    if (argc >= 7 && strcmp(argv[1], "bench") == 0) {
        method_fn mm = find_method(argv[2]);
        int n = atoi(argv[3]), warmups = atoi(argv[4]), reps = atoi(argv[5]);
        int block = argc >= 8 ? atoi(argv[7]) : 64;
        char path[1024];
        snprintf(path, sizeof path, "%s/A_%d.bin", argv[6], n);
        double *A = load(path, n);
        snprintf(path, sizeof path, "%s/B_%d.bin", argv[6], n);
        double *B = load(path, n);
        for (int r = 0; r < warmups + reps; r++) {
            double start = now();
            double *C = mm(A, B, n, block);
            double elapsed = now() - start;
            printf("%s %.9f %" PRId64 "\n", r < warmups ? "warmup" : "rep", elapsed, checksum_bits(C, n));
            fflush(stdout);
            free(C);
        }
        free(A);
        free(B);
    } else if (argc >= 7 && strcmp(argv[1], "multiply") == 0) {
        method_fn mm = find_method(argv[2]);
        int n = atoi(argv[3]);
        int block = argc >= 8 ? atoi(argv[7]) : 64;
        double *A = load(argv[4], n), *B = load(argv[5], n);
        double *C = mm(A, B, n, block);
        save(argv[6], C, n);
        free(A); free(B); free(C);
    } else {
        fprintf(stderr, "Usage: %s bench <method> <n> <warmups> <reps> <data_dir> [block]\n"
                        "       %s multiply <method> <n> <A> <B> <C> [block]\n", argv[0], argv[0]);
        return 1;
    }
    return 0;
}
