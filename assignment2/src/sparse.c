/*
 * Sparse matrix multiplication in CSR format (Assignment 2, Part B). All code is our own; no library is used.
 *
 * Build: gcc -O2 -std=c11 -o sparse sparse.c
 * Usage (A.csr and B.csr are files written by scripts/generate_inputs.py):
 *   sparse spgemm  <warmups> <reps> <A.csr> <B.csr>          CSR x CSR -> CSR (Gustavson's algorithm)
 *   sparse dense   <warmups> <reps> <A.csr> <B.csr> <block>  same matrices expanded to dense, tiled dense product
 *   sparse convert <warmups> <reps> <A.csr> <B.csr>          cost of converting A and B from dense to CSR
 *   sparse multiply <A.csr> <B.csr> <C.csr>                  one CSR product, written to a file (tests)
 * Each measured line is "<warmup|rep> <seconds> <checksum_bits> <nnz_of_C>".
 *
 * CSR file layout (little-endian): int64 n, int64 nnz, int64 row_ptr[n+1], int32 col[nnz], float64 val[nnz].
 * Column indices are sorted inside each row.
 *
 * Timing boundaries:
 *   spgemm  : only the multiplication of the two CSR matrices, including the allocation of the result and the
 *             sorting of the column indices of each result row. Reading the files is not timed.
 *   dense   : only the dense multiplication (same tiled kernel as dense.c). The CSR -> dense expansion is not timed.
 *   convert : only the dense -> CSR conversion of A and B (the dense matrices are built before the timer starts).
 *
 * For every C[i][j] the products are added in increasing k order, as in the dense methods, so the CSR and dense
 * results are bit-identical and the checksums of all methods can be compared directly.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int64_t n, nnz, cap;
    int64_t *row_ptr;
    int32_t *col;
    double *val;
} CSR;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "Out of memory (%zu bytes)\n", bytes); exit(2); }
    return p;
}

static void read_exact(void *dst, size_t size, size_t count, FILE *f, const char *path) {
    if (fread(dst, size, count, f) != count) { fprintf(stderr, "Cannot read %s\n", path); exit(1); }
}

static CSR load_csr(const char *path) {
    CSR M;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    read_exact(&M.n, sizeof M.n, 1, f, path);
    read_exact(&M.nnz, sizeof M.nnz, 1, f, path);
    M.cap = M.nnz;
    M.row_ptr = xmalloc(sizeof(int64_t) * (M.n + 1));
    M.col = xmalloc(sizeof(int32_t) * M.nnz);
    M.val = xmalloc(sizeof(double) * M.nnz);
    read_exact(M.row_ptr, sizeof(int64_t), M.n + 1, f, path);
    read_exact(M.col, sizeof(int32_t), M.nnz, f, path);
    read_exact(M.val, sizeof(double), M.nnz, f, path);
    fclose(f);
    return M;
}

static void save_csr(const char *path, const CSR *M) {
    FILE *f = fopen(path, "wb");
    fwrite(&M->n, sizeof M->n, 1, f);
    fwrite(&M->nnz, sizeof M->nnz, 1, f);
    fwrite(M->row_ptr, sizeof(int64_t), M->n + 1, f);
    fwrite(M->col, sizeof(int32_t), M->nnz, f);
    fwrite(M->val, sizeof(double), M->nnz, f);
    fclose(f);
}

static void free_csr(CSR *M) { free(M->row_ptr); free(M->col); free(M->val); }

static int cmp_int32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* Gustavson's row-by-row algorithm with a dense accumulator (SPA) of length n. */
static CSR spgemm(const CSR *A, const CSR *B) {
    int64_t n = A->n;
    CSR C;
    C.n = n;
    C.nnz = 0;
    C.cap = A->nnz + B->nnz + n;
    C.row_ptr = xmalloc(sizeof(int64_t) * (n + 1));
    C.col = xmalloc(sizeof(int32_t) * C.cap);
    C.val = xmalloc(sizeof(double) * C.cap);
    double *acc = xmalloc(sizeof(double) * n);
    int64_t *marker = xmalloc(sizeof(int64_t) * n);
    int32_t *touched = xmalloc(sizeof(int32_t) * n);
    for (int64_t j = 0; j < n; j++) marker[j] = -1;

    C.row_ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t count = 0;
        for (int64_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            int32_t k = A->col[p];
            double a = A->val[p];
            for (int64_t q = B->row_ptr[k]; q < B->row_ptr[k + 1]; q++) {
                int32_t j = B->col[q];
                if (marker[j] != i) {
                    marker[j] = i;
                    touched[count++] = j;
                    acc[j] = a * B->val[q];
                } else {
                    acc[j] += a * B->val[q];
                }
            }
        }
        qsort(touched, (size_t)count, sizeof(int32_t), cmp_int32);
        if (C.nnz + count > C.cap) {
            while (C.nnz + count > C.cap) C.cap *= 2;
            C.col = realloc(C.col, sizeof(int32_t) * C.cap);
            C.val = realloc(C.val, sizeof(double) * C.cap);
            if (!C.col || !C.val) { fprintf(stderr, "Out of memory\n"); exit(2); }
        }
        for (int64_t t = 0; t < count; t++) {
            C.col[C.nnz] = touched[t];
            C.val[C.nnz] = acc[touched[t]];
            C.nnz++;
        }
        C.row_ptr[i + 1] = C.nnz;
    }
    free(acc); free(marker); free(touched);
    return C;
}

static double *to_dense(const CSR *M) {
    double *D = calloc((size_t)(M->n * M->n), sizeof(double));
    if (!D) { fprintf(stderr, "Out of memory\n"); exit(2); }
    for (int64_t i = 0; i < M->n; i++)
        for (int64_t p = M->row_ptr[i]; p < M->row_ptr[i + 1]; p++)
            D[i * M->n + M->col[p]] = M->val[p];
    return D;
}

/* Dense -> CSR in two passes: count the nonzeros, then fill. */
static CSR from_dense(const double *D, int64_t n) {
    CSR M;
    M.n = n;
    M.nnz = 0;
    for (int64_t i = 0; i < n * n; i++) M.nnz += D[i] != 0.0;
    M.cap = M.nnz;
    M.row_ptr = xmalloc(sizeof(int64_t) * (n + 1));
    M.col = xmalloc(sizeof(int32_t) * M.nnz);
    M.val = xmalloc(sizeof(double) * M.nnz);
    int64_t p = 0;
    M.row_ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++)
            if (D[i * n + j] != 0.0) { M.col[p] = (int32_t)j; M.val[p] = D[i * n + j]; p++; }
        M.row_ptr[i + 1] = p;
    }
    return M;
}

/* Same tiled kernel as mm_tiled in dense.c. */
static double *dense_tiled(const double *A, const double *B, int64_t n, int64_t block) {
    double *C = calloc((size_t)(n * n), sizeof(double));
    if (!C) { fprintf(stderr, "Out of memory\n"); exit(2); }
    for (int64_t ii = 0; ii < n; ii += block)
        for (int64_t kk = 0; kk < n; kk += block)
            for (int64_t jj = 0; jj < n; jj += block) {
                int64_t imax = ii + block < n ? ii + block : n;
                int64_t kmax = kk + block < n ? kk + block : n;
                int64_t jmax = jj + block < n ? jj + block : n;
                for (int64_t i = ii; i < imax; i++) {
                    double *Ci = C + i * n;
                    for (int64_t k = kk; k < kmax; k++) {
                        double a = A[i * n + k];
                        const double *Bk = B + k * n;
                        for (int64_t j = jj; j < jmax; j++)
                            Ci[j] += a * Bk[j];
                    }
                }
            }
    return C;
}

static int64_t bits_of(double x) { int64_t b; memcpy(&b, &x, sizeof b); return b; }

/* Row-major sum of the stored values (zeros do not change the sum, so it equals the dense checksum). */
static int64_t checksum_csr(const CSR *M) {
    double total = 0.0;
    for (int64_t p = 0; p < M->nnz; p++) total += M->val[p];
    return bits_of(total);
}

static int64_t checksum_dense(const double *D, int64_t n, int64_t *nnz) {
    double total = 0.0;
    *nnz = 0;
    for (int64_t i = 0; i < n * n; i++) { total += D[i]; *nnz += D[i] != 0.0; }
    return bits_of(total);
}

static void print_line(int r, int warmups, double elapsed, int64_t checksum, int64_t nnz) {
    printf("%s %.9f %" PRId64 " %" PRId64 "\n", r < warmups ? "warmup" : "rep", elapsed, checksum, nnz);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "spgemm") == 0) {
        int warmups = atoi(argv[2]), reps = atoi(argv[3]);
        CSR A = load_csr(argv[4]), B = load_csr(argv[5]);
        for (int r = 0; r < warmups + reps; r++) {
            double start = now();
            CSR C = spgemm(&A, &B);
            double elapsed = now() - start;
            print_line(r, warmups, elapsed, checksum_csr(&C), C.nnz);
            free_csr(&C);
        }
    } else if (argc >= 7 && strcmp(argv[1], "dense") == 0) {
        int warmups = atoi(argv[2]), reps = atoi(argv[3]);
        int64_t block = atoll(argv[6]);
        CSR As = load_csr(argv[4]), Bs = load_csr(argv[5]);
        double *A = to_dense(&As), *B = to_dense(&Bs);
        int64_t n = As.n;
        free_csr(&As); free_csr(&Bs);
        for (int r = 0; r < warmups + reps; r++) {
            double start = now();
            double *C = dense_tiled(A, B, n, block);
            double elapsed = now() - start;
            int64_t nnz, checksum = checksum_dense(C, n, &nnz);
            print_line(r, warmups, elapsed, checksum, nnz);
            free(C);
        }
    } else if (argc >= 6 && strcmp(argv[1], "convert") == 0) {
        int warmups = atoi(argv[2]), reps = atoi(argv[3]);
        CSR As = load_csr(argv[4]), Bs = load_csr(argv[5]);
        double *A = to_dense(&As), *B = to_dense(&Bs);
        int64_t n = As.n;
        for (int r = 0; r < warmups + reps; r++) {
            double start = now();
            CSR A2 = from_dense(A, n), B2 = from_dense(B, n);
            double elapsed = now() - start;
            if (A2.nnz != As.nnz || B2.nnz != Bs.nnz) { fprintf(stderr, "Conversion mismatch\n"); return 3; }
            print_line(r, warmups, elapsed, checksum_csr(&A2), A2.nnz + B2.nnz);
            free_csr(&A2); free_csr(&B2);
        }
    } else if (argc >= 5 && strcmp(argv[1], "multiply") == 0) {
        CSR A = load_csr(argv[2]), B = load_csr(argv[3]);
        CSR C = spgemm(&A, &B);
        save_csr(argv[4], &C);
    } else {
        fprintf(stderr, "Usage: %s spgemm|dense|convert <warmups> <reps> <A.csr> <B.csr> [block]\n"
                        "       %s multiply <A.csr> <B.csr> <C.csr>\n", argv[0], argv[0]);
        return 1;
    }
    return 0;
}
