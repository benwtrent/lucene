package org.apache.lucene.sandbox.rabitq;

public class MatrixUtils {
    public static float[][] multiply(float[][] a, float[][] b) {
        int n = a.length;
        int dA = a[0].length;
        int dB = b.length;
        int dC = b[0].length;

        if (dA != dB) {
            throw new RuntimeException("Matrix dimensions mismatch");
        }

        float[][] C = new float[n][dC];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dC; j++) {
                for (int k = 0; k < dA; k++) {
                    C[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return C;
    }

    public static void multiplyElementWise(float[][] a, float[][] b) {
        assert a.length == b.length;
        assert a[0].length == b[0].length;

        for(int i = 0; i < a.length; i++) {
            for( int j = 0; j < a[0].length; j++) {
                a[i][j] *= b[i][j];
            }
        }
    }

    public static void divide(float[][] a, float divisor) {
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                a[i][j] /= divisor;
            }
        }
    }

    public static void normalize(float[][] a, float[][] norms, float infReplacement) {
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                float v = a[i][j] / norms[i][0];
                if (!Float.isFinite(v)) {
                    a[i][j] = infReplacement;
                } else {
                    a[i][j] = v;
                }
            }
        }
    }

    public static float distance(float[][] a, int startA, float[][] b, int startB) {
        float[] vectorA = a[startA];
        float[] vectorB = b[startB];
        return VectorUtils.squareDistance(vectorA, vectorB);
    }

    public static float distance(float[] a, float[] b) {
        if ( a.length != b.length ) {
            throw new RuntimeException("vector distance can only be done on two vectors of the same dimensions");
        }

        int dimensions = a.length;
        float dist = 0f;
        for(int i = 0; i < dimensions; i++) {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }

        return dist;
    }

    public static float[][] transpose(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }

    public static float[][] dotProduct(float[][] x, float[][] p) {
        int m = x.length;
        int n = x[0].length;
        int p_n = p[0].length;
        if (n != p.length) {
            throw new IllegalArgumentException("Matrices are not compatible for dot product");
        }
        float[][] result = new float[m][p_n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p_n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += x[i][k] * p[k][j];
                }
            }
        }
        return result;
    }

    public static float[][] indexInto(float[][] a, int[] indicies) {
        // FIXME: FUTURE - check error conditions
        // FIXME: FUTURE - speed this up Arrays.copyOf?
        assert a.length == indicies.length;
        float[][] subMatrix = new float[indicies.length][a[0].length];
        for(int i = 0; i < indicies.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                subMatrix[i][j] = a[indicies[i]][j];
            }
        }
        return subMatrix;
    }

    public static float[][] subset(float[][] a, int lastColumn) {
        // FIXME: FUTURE - check error conditions
        float[][] subMatrix = new float[a.length][lastColumn];
        for(int i = 0; i < a.length; i++) {
            for (int j = 0; j < lastColumn; j++) {
                subMatrix[i][j] = a[i][j];
            }
        }
        return subMatrix;
    }

    public static boolean[][] subset(boolean[][] a, int lastColumn) {
        // FIXME: FUTURE - check error conditions
        boolean[][] subMatrix = new boolean[a.length][lastColumn];
        for(int i = 0; i < a.length; i++) {
            for (int j = 0; j < lastColumn; j++) {
                subMatrix[i][j] = a[i][j];
            }
        }
        return subMatrix;
    }


    public static float[][] subtract(float[][] a, float[][] b) {
        // FIXME: FUTURE - check error conditions
        float[][] result = new float[a.length][a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[i].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }

        return result;
    }

    public static float[][] sumRows(float[][] a) {
        // FIXME: FUTURE - check error conditions
        float[][] result = new float[a.length][1];
        for(int i = 0; i < a.length; i++) {
            float rowSum = 0f;
            for(int j = 0; j < a[i].length; j++) {
                rowSum += a[i][j];
            }
            result[i][0] = rowSum;
        }

        return result;
    }

    public static boolean[][] greaterThan(float[][] a, float v) {
        // FIXME: FUTURE - check error conditions
        boolean[][] conditions = new boolean[a.length][a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                conditions[i][j] = a[i][j] > v;
            }
        }

        return conditions;
    }

    public static float[][] padColumns(float[][] a, int dimension) {
        int pad = Math.max(a[0].length, dimension);
        float[][] aPad = new float[a.length][pad];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < pad; j++) {
                if( j < a[0].length ) {
                    aPad[i][j] = a[i][j];
                } else {
                    aPad[i][j] = 0;
                }
            }
        }
        return aPad;
    }

    public static float[][] normsForRows(float[][] m) {
        float[][] normalized = new float[m.length][1];
        for(int h = 0; h < m.length; h++) {
            float[] vector = m[h];
            // Calculate magnitude/length of the vector
            double magnitude = 0;
            for (int i = 0; i < vector.length; i++) {
                magnitude += Math.pow(vector[i], 2);
            }
            magnitude = Math.sqrt(magnitude);

            // FIXME: FUTURE - not good; sometimes this needs to be 0
//            if (magnitude == 0) {
//                throw new IllegalArgumentException("Cannot normalize a vector of length zero.");
//            }

            normalized[h][0] = (float) magnitude;
        }

        return normalized;
    }

    public static float[][] replaceInfinite(float[][] a, float value) {
        // FIXME: FUTURE - handle errors
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                if (!Float.isFinite(a[i][j])) {
                    a[i][j] = value;
                }
            }
        }
        return a;
    }

    // FIXME: FUTURE - this could be shorts instead of floats
    public static float[][] asFloats(boolean[][] a) {
        // FIXME: FUTURE - error handling
        float[][] aAsInts = new float[a.length][a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                aAsInts[i][j] = (short) (a[i][j] ? 1 : -1);
            }
        }

        return aAsInts;
    }

    public static boolean[] flatten(boolean[][] a) {
        boolean[] aFlattened = new boolean[a.length * a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                aFlattened[i*(a[0].length)+j] = a[i][j];
            }
        }
        return aFlattened;
    }

    public static float[] flatten(float[][] a) {
        float[] aFlattened = new float[a.length * a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[0].length; j++) {
                aFlattened[i*(a[0].length)+j] = a[i][j];
            }
        }
        return aFlattened;
    }

    public static long[][] repackAsUInt64(boolean[][] binXP, int B) {
        boolean[] binXPFlattened = MatrixUtils.flatten(binXP);

        int totalValues = B >> 6;
        int vectorSize = (binXPFlattened.length / 64) / totalValues;

        long[] binaryData = new long[vectorSize * totalValues];

        int idx = 0;
        for(int h = 0; h < binXPFlattened.length / 64; h++) {
            long result = 0L;
            int q = 0;
            for (int i = 7; i >= 0; i--) {
                for (int j = 7; j >= 0; j--) {
                    if (binXPFlattened[h * 64 + i * 8 + j]) {
                        result |= (1L << q);
                    }
                    q++;
                }
            }

            binaryData[idx] = result;
            idx++;
        }

        long[][] allBinary = new long[vectorSize][totalValues];

        idx = 0;
        for(int i = 0; i < vectorSize; i++) {
            for(int j = 0; j < totalValues; j++) {
                allBinary[i][j] = binaryData[idx];
                idx++;
            }
        }

        return allBinary;
    }
}
