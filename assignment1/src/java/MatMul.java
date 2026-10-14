import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Basic triple-loop matrix multiplication in Java (Assignment 1).
 *
 * Usage:
 *   java MatMul bench <n> <warmups> <reps> <data_dir>
 *       Loads A_<n>.bin and B_<n>.bin, runs <warmups> unreported multiplications (JIT warm-up)
 *       and <reps> measured ones. Prints one line per multiplication:
 *       "<warmup|rep> <seconds> <checksum_bits>".
 *   java MatMul multiply <n> <A.bin> <B.bin> <C.bin>
 *       Multiplies the given matrices once and writes C (used by the correctness tests).
 *
 * Matrices are stored on disk as raw little-endian float64 values in row-major order.
 * In memory they are double[][] (an array of independent row arrays).
 */
public class MatMul {

    static double[][] load(String path, int n) throws IOException {
        DoubleBuffer buf = ByteBuffer.wrap(Files.readAllBytes(Paths.get(path)))
                .order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            buf.get(M[i]);
        return M;
    }

    static void save(String path, double[][] C) throws IOException {
        int n = C.length;
        ByteBuffer buf = ByteBuffer.allocate(8 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (double[] row : C)
            for (double x : row)
                buf.putDouble(x);
        Files.write(Paths.get(path), buf.array());
    }

    /** Classic i-j-k triple loop. Allocation of C is part of the timed kernel. */
    static double[][] multiply(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++)
                    s += A[i][k] * B[k][j];
                C[i][j] = s;
            }
        return C;
    }

    /** Sequential row-major sum of C, returned as the raw IEEE-754 bits of the double. */
    static long checksumBits(double[][] C) {
        double total = 0.0;
        for (double[] row : C)
            for (double x : row)
                total += x;
        return Double.doubleToRawLongBits(total);
    }

    public static void main(String[] args) throws IOException {
        String mode = args[0];
        if (mode.equals("bench")) {
            int n = Integer.parseInt(args[1]);
            int warmups = Integer.parseInt(args[2]);
            int reps = Integer.parseInt(args[3]);
            double[][] A = load(args[4] + "/A_" + n + ".bin", n);
            double[][] B = load(args[4] + "/B_" + n + ".bin", n);
            for (int r = 0; r < warmups + reps; r++) {
                long start = System.nanoTime();
                double[][] C = multiply(A, B, n);
                long end = System.nanoTime();
                String label = r < warmups ? "warmup" : "rep";
                System.out.println(label + " " + String.format(java.util.Locale.ROOT, "%.9f", (end - start) / 1e9)
                        + " " + checksumBits(C));
                System.out.flush();
            }
        } else if (mode.equals("multiply")) {
            int n = Integer.parseInt(args[1]);
            save(args[4], multiply(load(args[2], n), load(args[3], n), n));
        } else {
            throw new IllegalArgumentException("Unknown mode: " + mode);
        }
    }
}
