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

import java.util.Arrays;
import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestBinaryQuantization extends LuceneTestCase {

  public void testQuantizeForIndex() {
    int dimensions = random().nextInt(1, 4097);
    int discretizedDimensions = BQVectorUtils.discretize(dimensions, 64);

    int randIdx = random().nextInt(VectorSimilarityFunction.values().length);
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.values()[randIdx];

    BinaryQuantizer quantizer = new BinaryQuantizer(discretizedDimensions, similarityFunction);

    float[] centroid = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      centroid[i] = random().nextFloat(-50f, 50f);
    }

    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random().nextFloat(-50f, 50f);
    }

    byte[] destination = new byte[discretizedDimensions / 8];
    float[] corrections = quantizer.quantizeForIndex(vector, destination, centroid);

    for (float correction : corrections) {
      assertFalse(Float.isNaN(correction));
    }

    if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
      assertEquals(3, corrections.length);
      assertTrue(corrections[0] >= 0);
      assertTrue(corrections[1] > 0);
    } else {
      assertEquals(2, corrections.length);
      assertTrue(corrections[0] > 0);
      assertTrue(corrections[1] > 0);
    }
  }

  public void testQuantizeForQuery() {
    int dimensions = random().nextInt(1, 4097);
    int discretizedDimensions = BQVectorUtils.discretize(dimensions, 64);

    int randIdx = random().nextInt(VectorSimilarityFunction.values().length);
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.values()[randIdx];

    BinaryQuantizer quantizer = new BinaryQuantizer(discretizedDimensions, similarityFunction);

    float[] centroid = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      centroid[i] = random().nextFloat(-50f, 50f);
    }

    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random().nextFloat(-50f, 50f);
    }

    byte[] destination = new byte[discretizedDimensions / 8 * BQSpaceUtils.B_QUERY];
    BinaryQuantizer.QueryFactors corrections =
        quantizer.quantizeForQuery(vector, destination, centroid);

    if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
      int sumQ = corrections.quantizedSum();
      float distToC = corrections.distToC();
      float lower = corrections.lower();
      float width = corrections.width();
      float normVmC = corrections.normVmC();
      float vDotC = corrections.vDotC();
      float cDotC = corrections.cDotC();
      assertTrue(sumQ >= 0);
      assertTrue(distToC >= 0);
      assertFalse(Float.isNaN(lower));
      assertTrue(width >= 0);
      assertTrue(normVmC >= 0);
      assertFalse(Float.isNaN(vDotC));
      assertTrue(cDotC >= 0);
    } else {
      int sumQ = corrections.quantizedSum();
      float distToC = corrections.distToC();
      float lower = corrections.lower();
      float width = corrections.width();
      assertTrue(sumQ >= 0);
      assertTrue(distToC >= 0);
      assertFalse(Float.isNaN(lower));
      assertTrue(width >= 0);
      assertEquals(corrections.normVmC(), 0.0f, 0.01f);
      assertEquals(corrections.vDotC(), 0.0f, 0.01f);
      assertEquals(corrections.cDotC(), 0.0f, 0.01f);
    }
  }

  public void testQuantizeForIndexEuclidean() {
    int dimensions = 128;

    VectorSimilarityFunction[] similarityFunctionsActingLikeEucllidean =
        new VectorSimilarityFunction[] {
          VectorSimilarityFunction.EUCLIDEAN,
          VectorSimilarityFunction.COSINE,
          VectorSimilarityFunction.DOT_PRODUCT
        };
    int randIdx = random().nextInt(similarityFunctionsActingLikeEucllidean.length);
    VectorSimilarityFunction similarityFunction = similarityFunctionsActingLikeEucllidean[randIdx];

    BinaryQuantizer quantizer = new BinaryQuantizer(dimensions, similarityFunction);
    float[] vector =
        new float[] {
          0f, 0.0f, 16.0f, 35.0f, 5.0f, 32.0f, 31.0f, 14.0f, 10.0f, 11.0f, 78.0f, 55.0f, 10.0f,
          45.0f, 83.0f, 11.0f, 6.0f, 14.0f, 57.0f, 102.0f, 75.0f, 20.0f, 8.0f, 3.0f, 5.0f, 67.0f,
          17.0f, 19.0f, 26.0f, 5.0f, 0.0f, 1.0f, 22.0f, 60.0f, 26.0f, 7.0f, 1.0f, 18.0f, 22.0f,
          84.0f, 53.0f, 85.0f, 119.0f, 119.0f, 4.0f, 24.0f, 18.0f, 7.0f, 7.0f, 1.0f, 81.0f, 106.0f,
          102.0f, 72.0f, 30.0f, 6.0f, 0.0f, 9.0f, 1.0f, 9.0f, 119.0f, 72.0f, 1.0f, 4.0f, 33.0f,
          119.0f, 29.0f, 6.0f, 1.0f, 0.0f, 1.0f, 14.0f, 52.0f, 119.0f, 30.0f, 3.0f, 0.0f, 0.0f,
          55.0f, 92.0f, 111.0f, 2.0f, 5.0f, 4.0f, 9.0f, 22.0f, 89.0f, 96.0f, 14.0f, 1.0f, 0.0f,
          1.0f, 82.0f, 59.0f, 16.0f, 20.0f, 5.0f, 25.0f, 14.0f, 11.0f, 4.0f, 0.0f, 0.0f, 1.0f,
          26.0f, 47.0f, 23.0f, 4.0f, 0.0f, 0.0f, 4.0f, 38.0f, 83.0f, 30.0f, 14.0f, 9.0f, 4.0f, 9.0f,
          17.0f, 23.0f, 41.0f, 0.0f, 0.0f, 2.0f, 8.0f, 19.0f, 25.0f, 23.0f
        };
    byte[] destination = new byte[dimensions / 8];
    float[] centroid =
        new float[] {
          27.054054f, 22.252253f, 25.027027f, 23.55856f, 31.099098f, 28.765766f, 31.64865f,
          30.981981f, 24.675676f, 21.81982f, 26.72973f, 25.486486f, 30.504505f, 35.216217f,
          28.306307f, 24.486486f, 29.675676f, 26.153152f, 31.315315f, 25.225225f, 29.234234f,
          30.855856f, 24.495495f, 29.828829f, 31.54955f, 24.36937f, 25.108109f, 24.873875f,
          22.918919f, 24.918919f, 29.027027f, 25.513514f, 27.64865f, 28.405405f, 23.603603f,
          17.900902f, 22.522522f, 24.855856f, 31.396397f, 32.585587f, 26.297297f, 27.468468f,
          19.675676f, 19.018019f, 24.801802f, 30.27928f, 27.945946f, 25.324324f, 29.918919f,
          27.864864f, 28.081081f, 23.45946f, 28.828829f, 28.387388f, 25.387388f, 27.90991f,
          25.621622f, 21.585585f, 26.378378f, 24.144144f, 21.666666f, 22.72973f, 26.837837f,
          22.747747f, 29.0f, 28.414415f, 24.612612f, 21.594595f, 19.117117f, 24.045046f,
          30.612612f, 27.55856f, 25.117117f, 27.783783f, 21.639639f, 19.36937f, 21.252253f,
          29.153152f, 29.216217f, 24.747747f, 28.252253f, 25.288288f, 25.738739f, 23.44144f,
          24.423424f, 23.693693f, 26.306307f, 29.162163f, 28.684685f, 34.648647f, 25.576576f,
          25.288288f, 29.63063f, 20.225225f, 25.72973f, 29.009008f, 28.666666f, 29.243244f,
          26.36937f, 25.864864f, 21.522522f, 21.414415f, 25.963964f, 26.054054f, 25.099098f,
          30.477478f, 29.55856f, 24.837837f, 24.801802f, 21.18018f, 24.027027f, 26.360361f,
          33.153152f, 29.135136f, 30.486486f, 28.639639f, 27.576576f, 24.486486f, 26.297297f,
          21.774775f, 25.936937f, 35.36937f, 25.171171f, 30.405405f, 31.522522f, 29.765766f,
          22.324324f, 26.09009f
        };
    float[] corrections = quantizer.quantizeForIndex(vector, destination, centroid);

    assertEquals(2, corrections.length);
    float distToCentroid = corrections[0];
    float magnitude = corrections[1];

    assertEquals(387.90204f, distToCentroid, 0.0003f);
    assertEquals(0.75916624f, magnitude, 0.0000001f);
    assertArrayEquals(
        new byte[] {20, 54, 56, 72, 97, -16, 62, 12, -32, -29, -125, 12, 0, -63, -63, -126},
        destination);
  }

  public void testQuantizeForQueryEuclidean() {
    int dimensions = 128;

    VectorSimilarityFunction[] similarityFunctionsActingLikeEucllidean =
        new VectorSimilarityFunction[] {
          VectorSimilarityFunction.EUCLIDEAN,
          VectorSimilarityFunction.COSINE,
          VectorSimilarityFunction.DOT_PRODUCT
        };
    int randIdx = random().nextInt(similarityFunctionsActingLikeEucllidean.length);
    VectorSimilarityFunction similarityFunction = similarityFunctionsActingLikeEucllidean[randIdx];

    BinaryQuantizer quantizer = new BinaryQuantizer(dimensions, similarityFunction);
    float[] vector =
        new float[] {
          0.0f, 8.0f, 69.0f, 45.0f, 2.0f, 0f, 16.0f, 52.0f, 32.0f, 13.0f, 2.0f, 6.0f, 34.0f, 49.0f,
          45.0f, 83.0f, 6.0f, 2.0f, 26.0f, 57.0f, 14.0f, 46.0f, 19.0f, 9.0f, 4.0f, 13.0f, 53.0f,
          104.0f, 33.0f, 11.0f, 25.0f, 19.0f, 30.0f, 10.0f, 7.0f, 2.0f, 8.0f, 7.0f, 25.0f, 1.0f,
          2.0f, 25.0f, 24.0f, 28.0f, 61.0f, 83.0f, 41.0f, 9.0f, 14.0f, 3.0f, 7.0f, 114.0f, 114.0f,
          114.0f, 114.0f, 5.0f, 5.0f, 1.0f, 5.0f, 114.0f, 73.0f, 75.0f, 106.0f, 3.0f, 5.0f, 6.0f,
          6.0f, 8.0f, 15.0f, 45.0f, 2.0f, 15.0f, 7.0f, 114.0f, 103.0f, 6.0f, 5.0f, 4.0f, 9.0f,
          67.0f, 47.0f, 22.0f, 32.0f, 27.0f, 41.0f, 10.0f, 114.0f, 36.0f, 43.0f, 42.0f, 23.0f, 9.0f,
          7.0f, 30.0f, 114.0f, 19.0f, 7.0f, 5.0f, 6.0f, 6.0f, 21.0f, 48.0f, 2.0f, 1.0f, 0.0f, 8.0f,
          114.0f, 13.0f, 0.0f, 1.0f, 53.0f, 83.0f, 14.0f, 8.0f, 16.0f, 12.0f, 16.0f, 20.0f, 27.0f,
          87.0f, 45.0f, 50.0f, 15.0f, 5.0f, 5.0f, 6.0f, 32.0f, 49.0f
        };
    byte[] destination = new byte[dimensions / 8 * BQSpaceUtils.B_QUERY];
    float[] centroid =
        new float[] {
          26.7f, 16.2f, 10.913f, 10.314f, 12.12f, 14.045f, 15.887f, 16.864f, 32.232f, 31.567f,
          34.922f, 21.624f, 16.349f, 29.625f, 31.994f, 22.044f, 37.847f, 24.622f, 36.299f, 27.966f,
          14.368f, 19.248f, 30.778f, 35.927f, 27.019f, 16.381f, 17.325f, 16.517f, 13.272f, 9.154f,
          9.242f, 17.995f, 53.777f, 23.011f, 12.929f, 16.128f, 22.16f, 28.643f, 25.861f, 27.197f,
          59.883f, 40.878f, 34.153f, 22.795f, 24.402f, 37.427f, 34.19f, 29.288f, 61.812f, 26.355f,
          39.071f, 37.789f, 23.33f, 22.299f, 28.64f, 47.828f, 52.457f, 21.442f, 24.039f, 29.781f,
          27.707f, 19.484f, 14.642f, 28.757f, 54.567f, 20.936f, 25.112f, 25.521f, 22.077f, 18.272f,
          14.526f, 29.054f, 61.803f, 24.509f, 37.517f, 35.906f, 24.106f, 22.64f, 32.1f, 48.788f,
          60.102f, 39.625f, 34.766f, 22.497f, 24.397f, 41.599f, 38.419f, 30.99f, 55.647f, 25.115f,
          14.96f, 18.882f, 26.918f, 32.442f, 26.231f, 27.107f, 26.828f, 15.968f, 18.668f, 14.071f,
          10.906f, 8.989f, 9.721f, 17.294f, 36.32f, 21.854f, 35.509f, 27.106f, 14.067f, 19.82f,
          33.582f, 35.997f, 33.528f, 30.369f, 36.955f, 21.23f, 15.2f, 30.252f, 34.56f, 22.295f,
          29.413f, 16.576f, 11.226f, 10.754f, 12.936f, 15.525f, 15.868f, 16.43f
        };
    BinaryQuantizer.QueryFactors corrections =
        quantizer.quantizeForQuery(vector, destination, centroid);

    int sumQ = corrections.quantizedSum();
    float lower = corrections.lower();
    float width = corrections.width();

    assertEquals(729, sumQ);
    assertEquals(-57.883f, lower, 0.001f);
    assertEquals(9.972266f, width, 0.000001f);
    assertArrayEquals(
        new byte[] {
          -77, -49, 73, -17, -89, 9, -43, -27, 40, 15, 42, 76, -122, 38, -22, -37, -96, 111, -63,
          -102, -123, 23, 110, 127, 32, 95, 29, 106, -120, -121, -32, -94, 78, -98, 42, 95, 122,
          114, 30, 18, 91, 97, -5, -9, 123, 122, 31, -66, 49, 1, 20, 48, 0, 12, 30, 30, 4, 96, 2, 2,
          4, 33, 1, 65
        },
        destination);
  }

  private float[] generateRandomFloatArray(
      Random random, int dimensions, float lowerBoundInclusive, float upperBoundExclusive) {
    float[] data = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      data[i] = random.nextFloat(lowerBoundInclusive, upperBoundExclusive);
    }
    return data;
  }

  public void testQuantizeForIndexMIP() {
    int dimensions = 768;

    // we want fixed values for these arrays so define our own random generation here to track
    // quantization changes
    Random random = new Random(42);

    float[] mipVectorToIndex = generateRandomFloatArray(random, dimensions, -1f, 1f);
    float[] mipCentroid = generateRandomFloatArray(random, dimensions, -1f, 1f);

    BinaryQuantizer quantizer =
        new BinaryQuantizer(dimensions, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    float[] vector = mipVectorToIndex;
    byte[] destination = new byte[dimensions / 8];
    float[] centroid = mipCentroid;
    float[] corrections = quantizer.quantizeForIndex(vector, destination, centroid);

    assertEquals(3, corrections.length);
    float ooq = corrections[0];
    float normOC = corrections[1];
    float oDotC = corrections[2];

    assertEquals(0.8141399f, ooq, 0.0000001f);
    assertEquals(21.847124f, normOC, 0.000001f);
    assertEquals(6.4300356f, oDotC, 0.0001f);
    assertArrayEquals(
        new byte[] {
          -83, -91, -71, 97, 32, -96, 89, -80, -19, -108, 3, 113, -111, 12, -86, 32, -43, 76, 122,
          -106, -83, -37, -122, 118, 84, -72, 34, 20, 57, -29, 119, -8, -10, -100, -109, 62, -54,
          53, -44, 8, -16, 80, 58, 50, 105, -25, 47, 115, -106, -92, -122, -44, 8, 18, -23, 24, -15,
          62, 58, 111, 99, -116, -111, -5, 101, -69, -32, -74, -105, 113, -89, 44, 100, -93, -80,
          82, -64, 91, -87, -95, 115, 6, 76, 110, 101, 39, 108, 72, 2, 112, -63, -43, 105, -42, 9,
          -128
        },
        destination);
  }

  public void testQuantizeForQueryMIP() {
    int dimensions = 768;

    // we want fixed values for these arrays so define our own random generation here to track
    // quantization changes
    Random random = new Random(42);

    float[] mipVectorToQuery = generateRandomFloatArray(random, dimensions, -1f, 1f);
    float[] mipCentroid = generateRandomFloatArray(random, dimensions, -1f, 1f);

    BinaryQuantizer quantizer =
        new BinaryQuantizer(dimensions, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    float[] vector = mipVectorToQuery;
    byte[] destination = new byte[dimensions / 8 * BQSpaceUtils.B_QUERY];
    float[] centroid = mipCentroid;
    BinaryQuantizer.QueryFactors corrections =
        quantizer.quantizeForQuery(vector, destination, centroid);

    int sumQ = corrections.quantizedSum();
    float lower = corrections.lower();
    float width = corrections.width();
    float normVmC = corrections.normVmC();
    float vDotC = corrections.vDotC();
    float cDotC = corrections.cDotC();

    System.out.println(Arrays.toString(destination));
    assertEquals(5272, sumQ);
    assertEquals(-0.08603752f, lower, 0.00000001f);
    assertEquals(0.011431276f, width, 0.00000001f);
    assertEquals(21.847124f, normVmC, 0.000001f);
    assertEquals(6.4300356f, vDotC, 0.0001f);
    assertEquals(252.37146f, cDotC, 0.0001f);
    assertArrayEquals(
        new byte[] {
          -81, 19, 67, 33, 112, 8, 40, -5, -19, 115, -87, -63, -59, 12, -2, -127, -23, 43, 24, 16,
          -69, 112, -22, 75, -81, -50, 100, -41, 3, -120, -93, -4, 4, 125, 34, -57, -109, 89, -63,
          -35, -116, 4, 35, 93, -26, -88, -55, -86, 63, -46, -122, -96, -26, 124, -64, 21, 96, 46,
          98, 97, 88, -98, -83, 121, 16, -14, -89, -118, 65, -39, -111, -35, 113, 108, 111, 86, 17,
          -69, -47, 72, 1, 36, 17, 113, -87, -5, -46, -37, -2, 93, -123, 118, 4, -12, -33, 95, 32,
          -63, -97, -109, 27, 111, 42, -57, -87, -41, -73, -106, 27, -31, 32, -1, 9, -88, -35, -11,
          -103, 5, 27, -127, 108, 127, -119, 58, 38, 18, -103, -27, -63, 56, 77, -13, 3, -40, -127,
          37, 82, -87, -26, -45, -14, 18, -50, 76, 25, 37, -12, 106, 17, 115, 0, 23, -109, 26, -110,
          17, -35, 111, 4, 60, 58, -64, -104, -125, 23, -58, 89, -117, 104, -71, 3, -89, -26, 46,
          15, 82, -83, -75, -72, -69, 20, -38, -47, 109, -66, -66, -89, 108, -122, -3, -69, -85, 18,
          59, 85, -97, -114, 95, 2, -84, -77, 121, -6, 10, 110, -13, -123, -34, 106, -71, -107, 123,
          67, -111, 58, 52, -53, 87, -113, -21, -44, 26, 10, -62, 56, 111, 36, -126, 26, 94, -88,
          -13, -113, -50, -9, -115, 84, 8, -32, -102, -4, 89, 29, 75, -73, -19, 22, -90, 76, -61, 4,
          -48, -100, -11, 107, 20, -39, -98, 123, 77, 104, 9, 9, 91, -105, -40, -106, -87, 38, 48,
          60, 29, -68, 124, -78, -63, -101, -115, 67, -17, 101, -53, 121, 44, -78, -12, 110, 91,
          -83, -92, -72, 96, 32, -96, 89, 48, 76, -124, 3, 113, -111, 12, -86, 32, -43, 68, 106,
          -122, -84, -37, -124, 118, 84, -72, 34, 20, 57, -29, 119, 56, -10, -108, -109, 60, -56,
          37, 84, 8, -16, 80, 24, 50, 41, -25, 47, 115, -122, -92, -126, -44, 8, 18, -23, 24, -15,
          60, 58, 111, 99, -120, -111, -21, 101, 59, -32, -74, -105, 113, -90, 36, 100, -93, -80,
          82, -64, 91, -87, -95, 115, 6, 76, 110, 101, 39, 44, 0, 2, 112, -64, -47, 105, 2, 1, -128
        },
        destination);
  }
}
