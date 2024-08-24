/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.util.quantization;

import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;

public class BinaryQuantizer {
  private final int discretizedDimensions;

  // dim floats random numbers sampled from the uniform distribution [0,1]
  private final float[] uniformDistribution;

  private final VectorSimilarityFunction similarityFunction;
  private final float sqrtDimensions;

  BinaryQuantizer(
      int dimensions,
      VectorSimilarityFunction similarityFunction,
      boolean fixedUniformDistribution) {
    if (dimensions <= 0) {
      throw new IllegalArgumentException("dimensions must be > 0 but was: " + dimensions);
    }
    this.discretizedDimensions = BQVectorUtils.discretize(dimensions, 64);
    this.similarityFunction = similarityFunction;
    Random random = new Random(42);
    uniformDistribution = new float[discretizedDimensions];
    for (int i = 0; i < discretizedDimensions; i++) {
      if (fixedUniformDistribution) {
        uniformDistribution[i] = 0.5f;
      } else {
        uniformDistribution[i] = (float) random.nextDouble();
      }
    }
    this.sqrtDimensions = (float) Math.sqrt(discretizedDimensions);
  }

  public BinaryQuantizer(int dimensions, VectorSimilarityFunction similarityFunction) {
    this(dimensions, similarityFunction, false);
  }

  private static float[] subset(float[] a, int lastColumn) {
    if (a.length == lastColumn) {
      return a;
    }
    return ArrayUtil.copyOfSubArray(a, 0, lastColumn);
  }

  private static void removeSignAndDivide(float[] a, float divisor) {
    for (int i = 0; i < a.length; i++) {
      a[i] = Math.abs(a[i]) / divisor;
    }
  }

  private static float sumAndNormalize(float[] a, float norm) {
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

  private static byte[] packAsBinary(float[] vector, int dimensions) {
    int totalValues = dimensions / 8;

    byte[] allBinary = new byte[totalValues];

    for (int h = 0; h < vector.length; h += 8) {
      byte result = 0;
      int q = 0;
      for (int i = 7; i >= 0; i--) {
        if (vector[h + i] > 0) {
          result |= (byte) (1 << q);
        }
        q++;
      }
      allBinary[h / 8] = result;
    }

    return allBinary;
  }

  public VectorSimilarityFunction getSimilarity() {
    return this.similarityFunction;
  }

  private record SubspaceOutput(byte[] packedBinaryVector, float projection) {}

  private SubspaceOutput generateSubSpace(float[] vector, float[] centroid) {
    // typically no-op if dimensions/64
    float[] paddedCentroid = BQVectorUtils.pad(centroid, discretizedDimensions);
    float[] paddedVector = BQVectorUtils.pad(vector, discretizedDimensions);

    BQVectorUtils.subtractInPlace(paddedVector, paddedCentroid);

    // The inner product between the data vector and the quantized data vector
    float norm = BQVectorUtils.norm(paddedVector);

    byte[] packedBinaryVector = packAsBinary(paddedVector, discretizedDimensions);

    paddedVector = subset(paddedVector, discretizedDimensions); // typically no-op if dimensions/64
    removeSignAndDivide(paddedVector, (float) Math.sqrt(discretizedDimensions));
    float projection = sumAndNormalize(paddedVector, norm);

    return new SubspaceOutput(packedBinaryVector, projection);
  }

  record SubspaceOutputMIP(
      byte[] packedBinaryVector, float OOQ, float normOC, float oDotC) {}

  private SubspaceOutputMIP generateSubSpaceMIP(float[] vector, float[] centroid) {

    // typically no-op if dimensions/64
    float[] paddedCentroid = BQVectorUtils.pad(centroid, discretizedDimensions);
    float[] paddedVector = BQVectorUtils.pad(vector, discretizedDimensions);

    float oDotC = VectorUtil.dotProduct(paddedVector, paddedCentroid);
    BQVectorUtils.subtractInPlace(paddedVector, paddedCentroid);

    float normOC = BQVectorUtils.norm(paddedVector);
    float[] normOMinusC = BQVectorUtils.divide(paddedVector, normOC); // OmC / norm(OmC)

    byte[] packedBinaryVector = packAsBinary(paddedVector, discretizedDimensions);

    float OOQ = computerOOQ(vector, normOMinusC, packedBinaryVector);

    return new SubspaceOutputMIP(packedBinaryVector, OOQ, normOC, oDotC);
  }

  private float computerOOQ(float[] vector, float[] normOMinusC, byte[] packedBinaryVector) {
    float OOQ = 0f;
    for (int j = 0; j < vector.length / 8; j++) {
      for (int r = 0; r < 8; r++) {
        int sign = ((packedBinaryVector[j] >> (7 - r)) & 0b00000001);
        OOQ += (normOMinusC[j * 8 + r] * (2 * sign - 1));
      }
    }
    OOQ = OOQ / sqrtDimensions;
    return OOQ;
  }

  private static float[] range(float[] q) {
    float vl = 1e20f;
    float vr = -1e20f;
    for (int i = 0; i < q.length; i++) {
      if (q[i] < vl) {
        vl = q[i];
      }
      if (q[i] > vr) {
        vr = q[i];
      }
    }

    return new float[] {vl, vr};
  }

  public float[] quantizeForIndex(float[] vector, byte[] destination, float[] centroid) {
    assert this.discretizedDimensions == BQVectorUtils.discretize(vector.length, 64);

    if (this.discretizedDimensions != destination.length * 8) {
      throw new IllegalArgumentException(
          "vector and quantized vector destination must be compatible dimensions: "
              + BQVectorUtils.discretize(vector.length, 64)
              + " [ "
              + this.discretizedDimensions
              + " ]"
              + "!= "
              + destination.length
              + " * 8");
    }

    if (vector.length != centroid.length) {
      throw new IllegalArgumentException(
          "vector and centroid dimensions must be the same: "
              + vector.length
              + "!= "
              + centroid.length);
    }

    float[] corrections;

    // FIXME: make a copy of vector so we don't overwrite it here?
    //  ... (could trade subtractInPlace w subtract in genSubSpace)
    vector = ArrayUtil.copyArray(vector);

    switch (similarityFunction) {
      case VectorSimilarityFunction.EUCLIDEAN:
      case VectorSimilarityFunction.COSINE:
      case VectorSimilarityFunction.DOT_PRODUCT:
        float distToCentroid = (float) Math.sqrt(VectorUtil.squareDistance(vector, centroid));

        SubspaceOutput subspaceOutput = generateSubSpace(vector, centroid);
        corrections = new float[2];
        // FIXME: quantize these values so we are passing back 1 byte values for all three of these
        corrections[0] = distToCentroid;
        corrections[1] = subspaceOutput.projection();
        System.arraycopy(
            subspaceOutput.packedBinaryVector(), 0, destination, 0, destination.length);
        break;
      case VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT:
        SubspaceOutputMIP subspaceOutputMIP = generateSubSpaceMIP(vector, centroid);
        corrections = new float[3];
        // FIXME: quantize these values so we are passing back 1 byte values for all three of these
        corrections[0] = subspaceOutputMIP.OOQ();
        corrections[1] = subspaceOutputMIP.normOC();
        corrections[2] = subspaceOutputMIP.oDotC();
        System.arraycopy(
            subspaceOutputMIP.packedBinaryVector(), 0, destination, 0, destination.length);
        break;
      default:
        throw new UnsupportedOperationException(
            "Unsupported similarity function: " + similarityFunction);
    }

    return corrections;
  }

  private record QuantResult(byte[] result, short quantizedSum) {}

  private static QuantResult quantize(
      float[] vector, float[] uniformRand, float lower, float width) {
    // FIXME: speed up with panama? and/or use existing scalar quantization utils in Lucene?
    byte[] result = new byte[vector.length];
    float oneOverWidth = 1.0f / width;
    short sumQ = 0;
    for (int i = 0; i < vector.length; i++) {
      byte res = (byte) ((vector[i] - lower) * oneOverWidth + uniformRand[i]);
      result[i] = res;
      sumQ += res;
    }

    return new QuantResult(result, sumQ);
  }

  public record QueryFactors(
      short quantizedSum, float lower, float width, float normVmC, float vDotC, float cDotC) {}

  public QueryFactors quantizeForQuery(float[] vector, byte[] destination, float[] centroid) {
    assert this.discretizedDimensions == BQVectorUtils.discretize(vector.length, 64);

    if (this.discretizedDimensions != (destination.length * 8) / BQSpaceUtils.B_QUERY) {
      throw new IllegalArgumentException(
          "vector and quantized vector destination must be compatible dimensions: "
              + BQVectorUtils.discretize(vector.length, 64)
              + " [ "
              + this.discretizedDimensions
              + " ]"
              + "!= ("
              + destination.length
              + " * 8) / "
              + BQSpaceUtils.B_QUERY);
    }

    if (vector.length != centroid.length) {
      throw new IllegalArgumentException(
          "vector and centroid dimensions must be the same: "
              + vector.length
              + "!= "
              + centroid.length);
    }

    // FIXME: make a copy of vector so we don't overwrite it here?
    //  ... (could subtractInPlace but the passed vector is modified)
    float[] vmC = BQVectorUtils.subtract(vector, centroid);

    // FIXME: should other similarity functions behave like MIP on query like COSINE
    float normVmC = 0f;
    if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
      normVmC = BQVectorUtils.norm(vmC);
      vmC = BQVectorUtils.divide(vmC, normVmC);
    }
    float[] range = range(vmC);
    float lower = range[0];
    float upper = range[1];
    // Δ := (𝑣𝑟 − 𝑣𝑙)/(2𝐵𝑞 − 1)
    float width = (upper - lower) / ((1 << BQSpaceUtils.B_QUERY) - 1);

    QuantResult quantResult = quantize(vmC, uniformDistribution, lower, width);
    byte[] byteQuery = quantResult.result();

    // q¯ = Δ · q¯𝑢 + 𝑣𝑙 · 1𝐷
    // q¯ is an approximation of q′  (scalar quantized approximation)
    // FIXME: vectors need to be padded but that's expensive; update transponseBin to deal
    byteQuery = BQVectorUtils.pad(byteQuery, discretizedDimensions);
    BQSpaceUtils.transposeBin(byteQuery, discretizedDimensions, destination);

    QueryFactors factors;
    if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
      float vDotC = VectorUtil.dotProduct(vector, centroid);
      float cDotC = VectorUtil.dotProduct(centroid, centroid);
      // FIXME: quantize the corrections as well so we store less
      factors = new QueryFactors(quantResult.quantizedSum, lower, width, normVmC, vDotC, cDotC);
    } else {
      // FIXME: quantize the corrections as well so we store less
      factors = new QueryFactors(quantResult.quantizedSum, lower, width, 0f, 0f, 0f);
    }

    return factors;
  }
}
