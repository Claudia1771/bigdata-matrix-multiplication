/*
 * Sequential and OpenMP-parallel blocked matrix multiplication (Assignment 3).
 *
 * Build: gcc -O2 -std=c11 -fopenmp -o parallel parallel.c
 * Usage:
 *   parallel bench <method> <n> <threads> <warmups> <reps> <data_dir> [block] [schedule]
 *       Loads A_<n>.bin and B_<n>.bin, runs <warmups> unreported and <reps> measured multiplications.
 *       Prints one line per multiplication: "<warmup|rep> <seconds> <checksum_bits> <team_size>".
 *   parallel multiply <method> <n> <threads> <A.bin> <B.bin> <C.bin> [block] [schedule]
 *       One multiplication, used by the correctness tests.
 *
 * Methods:
 *   seq  the sequential blocked (tiled) method of Assignment 2, unchanged: the baseline T_seq
 *   par  the same blocked loops; the blocks of rows of C (loop ii) are distributed among the OpenMP threads.
 *        Each thread writes only its own rows of C, and every C[i][j] still adds its products in increasing
 *        k order, so no synchronisation is needed and the result is bit-identical to seq.
 * schedule: "dynamic" (default; one block row at a time, which adapts to cores of different speed) or "static".
 *
 * Matrices are raw little-endian float64 files in row-major order, stored as contiguous arrays.
 */
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Monotonic high-resolution timer (omp_get_wtime has only millisecond resolution on this platform). */
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

/* All the work for one block row of C (rows ii .. ii+block-1): the kk and jj block loops and the i-k-j loops
 * inside each block, exactly as in the tiled method of Assignment 2. Both methods call this same function, which
 * is kept out of line so that the compiler generates (and vectorises) a single copy of the kernel: the methods then
 * differ only in how the block rows are distributed. `restrict` states that C, freshly allocated, never overlaps A
 * or B; this is what lets the compiler vectorise the j loop, as it did in Assignment 2 when C was a local variable. */
__attribute__((noinline))
static void block_row(const double *restrict A, const double *restrict B, double *restrict C, int n, int block, int ii) {
    int imax = ii + block < n ? ii + block : n;
    for (int kk = 0; kk < n; kk += block) {
        int kmax = kk + block < n ? kk + block : n;
        for (int jj = 0; jj < n; jj += block) {
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
    }
}

/* Sequential baseline: the tiled method of Assignment 2 (block rows processed in order). */
static double *mm_seq(const double *A, const double *B, int n, int block, int dynamic_schedule) {
    (void)dynamic_schedule;
    double *C = calloc((size_t)n * n, sizeof(double));
    for (int ii = 0; ii < n; ii += block)
        block_row(A, B, C, n, block, ii);
    return C;
}

/* Parallel version: the block rows of C are distributed among the threads of the team. */
static double *mm_par(const double *A, const double *B, int n, int block, int dynamic_schedule) {
    double *C = calloc((size_t)n * n, sizeof(double));
    int nblocks = (n + block - 1) / block;
    if (dynamic_schedule) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < nblocks; b++)
            block_row(A, B, C, n, block, b * block);
    } else {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < nblocks; b++)
            block_row(A, B, C, n, block, b * block);
    }
    return C;
}

typedef double *(*method_fn)(const double *, const double *, int, int, int);

static method_fn find_method(const char *name) {
    if (strcmp(name, "seq") == 0) return mm_seq;
    if (strcmp(name, "par") == 0) return mm_par;
    fprintf(stderr, "Unknown method: %s\n", name);
    exit(1);
}

static int64_t checksum_bits(const double *C, int n) {
    double total = 0.0;
    for (long i = 0; i < (long)n * n; i++)
        total += C[i];
    int64_t bits;
    memcpy(&bits, &total, sizeof bits);
    return bits;
}

/* Number of threads that a parallel region really gets (OMP_DYNAMIC is disabled). */
static int team_size(void) {
    int size = 1;
    #pragma omp parallel
    {
        #pragma omp single
        size = omp_get_num_threads();
    }
    return size;
}

int main(int argc, char **argv) {
    omp_set_dynamic(0);
    if (argc >= 8 && strcmp(argv[1], "bench") == 0) {
        method_fn mm = find_method(argv[2]);
        int n = atoi(argv[3]), threads = atoi(argv[4]), warmups = atoi(argv[5]), reps = atoi(argv[6]);
        int block = argc >= 9 ? atoi(argv[8]) : 64;
        int dynamic_schedule = !(argc >= 10 && strcmp(argv[9], "static") == 0);
        omp_set_num_threads(threads);
        int team = strcmp(argv[2], "seq") == 0 ? 1 : team_size();
        char path[1024];
        snprintf(path, sizeof path, "%s/A_%d.bin", argv[7], n);
        double *A = load(path, n);
        snprintf(path, sizeof path, "%s/B_%d.bin", argv[7], n);
        double *B = load(path, n);
        for (int r = 0; r < warmups + reps; r++) {
            double start = now();
            double *C = mm(A, B, n, block, dynamic_schedule);
            double elapsed = now() - start;
            printf("%s %.9f %" PRId64 " %d\n", r < warmups ? "warmup" : "rep", elapsed, checksum_bits(C, n), team);
            fflush(stdout);
            free(C);
        }
        free(A);
        free(B);
    } else if (argc >= 8 && strcmp(argv[1], "multiply") == 0) {
        method_fn mm = find_method(argv[2]);
        int n = atoi(argv[3]), threads = atoi(argv[4]);
        int block = argc >= 9 ? atoi(argv[8]) : 64;
        int dynamic_schedule = !(argc >= 10 && strcmp(argv[9], "static") == 0);
        omp_set_num_threads(threads);
        double *A = load(argv[5], n), *B = load(argv[6], n);
        double *C = mm(A, B, n, block, dynamic_schedule);
        save(argv[7], C, n);
        free(A); free(B); free(C);
    } else {
        fprintf(stderr, "Usage: %s bench <seq|par> <n> <threads> <warmups> <reps> <data_dir> [block] [static|dynamic]\n"
                        "       %s multiply <seq|par> <n> <threads> <A> <B> <C> [block] [static|dynamic]\n", argv[0], argv[0]);
        return 1;
    }
    return 0;
}
