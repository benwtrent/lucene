package org.apache.lucene.sandbox.rabitq;

import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class MatrixUtils {
  // FIXME: check errors

  private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

  public static void partialRemoveSignAndDivide(float[] a, float divisor) {
    // FIXME: revert to old behavior for small dimensions
    //            for(int j = 0; j < a[0].length; j++) {
    //                a[i][j] = Math.abs(a[i][j]) / divisor;
    //            }
    int size = a.length / FLOAT_SPECIES.length();
    for (int r = 0; r < size; r++) {
      int offset = FLOAT_SPECIES.length() * r;
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, a, offset);
      va.abs().div(divisor).intoArray(a, offset);
    }

    // tail
    int remainder = a.length % FLOAT_SPECIES.length();
    if (remainder != 0) {
      for (int i = a.length - remainder; i < a.length; i++) {
        a[i] = Math.abs(a[i]) / divisor;
      }
    }
  }

  public static float partialSumAndNormalize(float[] a, float norm) {
    float aDivided = 0f;

    for (int i = 0; i < a.length; i++) {
      aDivided += a[i];
    }

    aDivided = aDivided / norm;
    if (!Float.isFinite(aDivided)) {
      aDivided = 0.8f; // can be anything
    }

    return aDivided;
  }

  public static float distance(float[][] a, int startA, float[][] b, int startB) {
    float[] vectorA = a[startA];
    float[] vectorB = b[startB];
    return VectorUtils.squareDistance(vectorA, vectorB);
  }

  public static float distance(float[][] a, int startA, float[] vectorB) {
    float[] vectorA = a[startA];
    return VectorUtils.squareDistance(vectorA, vectorB);
  }

  public static void transpose(float[][] a) {
    int m = a.length;
    int n = a[0].length;
    for (int i = 0; i < m; i++) {
      for (int j = i + 1; j < n; j++) {
        if (i != j) {
          float tmp = a[i][j];
          a[i][j] = a[j][i];
          a[j][i] = tmp;
        }
      }
    }
  }

  public static float[] partialSubset(float[] a, int lastColumn) {
    return Arrays.copyOf(a, lastColumn);
  }

  public static void partialSubtract(float[] a, float[] b) {
    for (int j = 0; j < a.length; j++) {
      a[j] -= b[j];
    }
  }

  public static float[][] padColumns(float[][] a, int dimension) {
    if (dimension == 0) {
      return a;
    }
    if (a[0].length == dimension) {
      return a;
    }
    int pad = Math.max(a[0].length, dimension);
    float[][] aPad = new float[a.length][pad];
    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < pad; j++) {
        if (j < a[0].length) {
          aPad[i][j] = a[i][j];
        } else {
          aPad[i][j] = 0;
        }
      }
    }
    return aPad;
  }

  public static float[] partialPadColumns(float[] a, int dimension) {
    if (dimension == 0) {
      return a;
    }
    if (a.length == dimension) {
      return a;
    }
    int pad = Math.max(a.length, dimension);
    float[] aPad = new float[pad];
    for (int j = 0; j < pad; j++) {
      if (j < a.length) {
        aPad[j] = a[j];
      } else {
        aPad[j] = 0;
      }
    }
    return aPad;
  }

  public static float partialNormForRow(float[] vector) {
    float normalized = 0f;
    // Calculate magnitude/length of the vector
    double magnitude = 0;

    int size = vector.length / FLOAT_SPECIES.length();
    for (int r = 0; r < size; r++) {
      int offset = FLOAT_SPECIES.length() * r;
      FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, vector, offset);
      magnitude += va.mul(va).reduceLanes(VectorOperators.ADD);
    }

    // tail
    int remainder = vector.length % FLOAT_SPECIES.length();
    if (remainder != 0) {
      for (int i = vector.length - remainder; i < vector.length; i++) {
        magnitude = Math.fma(vector[i], vector[i], magnitude);
      }
    }

    // FIXME: evaluate for small dimensions whether this is faster
    //            for (int i = 0; i < vector.length; i++) {
    //                magnitude = Math.fma(vector[i], vector[i], magnitude);
    //            }

    magnitude = Math.sqrt(magnitude);

    // FIXME: FUTURE - not good; sometimes this needs to be 0
    //            if (magnitude == 0) {
    //                throw new IllegalArgumentException("Cannot normalize a vector of length
    // zero.");
    //            }

    normalized = (float) magnitude;

    return normalized;
  }

  public static byte[] partialRepackAsUInt64(float[] XP, int B) {
    int totalValues = B / 8;

    byte[] allBinary = new byte[totalValues];

    for (int h = 0; h < XP.length; h += 8) {
      byte result = 0;
      int q = 0;
      for (int i = 7; i >= 0; i--) {
        if (XP[h + i] > 0) {
          result |= (byte) (1 << q);
        }
        q++;
      }
      allBinary[h / 8] = result;
    }

    return allBinary;
  }
}
