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

package org.apache.lucene.codecs.lucene912;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.BQSpaceUtils;
import org.apache.lucene.util.quantization.BQVectorUtils;
import org.apache.lucene.util.quantization.BinaryQuantizer;

public class TestLucene912BinaryFlatVectorsScorer extends LuceneTestCase {

  public void testScore() throws IOException {
    int dimensions = random().nextInt(1, 4097);
    int discretizedDimensions = BQVectorUtils.discretize(dimensions, 64);

    int randIdx = random().nextInt(VectorSimilarityFunction.values().length);
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.values()[randIdx];

    float[][] centroids = new float[2][dimensions];
    for (int i = 0; i < centroids.length; i++) {
      for (int j = 0; j < dimensions; j++) {
        centroids[i][j] = random().nextFloat(-50f, 50f);
      }
      if (similarityFunction == VectorSimilarityFunction.COSINE) {
        centroids[i] = VectorUtil.l2normalize(centroids[i]);
      }
    }

    Lucene912BinaryFlatVectorsScorer.BinaryQueryVector[] queryVectors =
        new Lucene912BinaryFlatVectorsScorer.BinaryQueryVector[2];
    for (int i = 0; i < 2; i++) {
      byte[] vector = new byte[discretizedDimensions / 8 * BQSpaceUtils.B_QUERY];
      random().nextBytes(vector);
      float distanceToCentroid = random().nextFloat(0f, 10_000.0f);
      float vl = random().nextFloat(-1000f, 1000f);
      float width = random().nextFloat(0f, 1000f);
      short quantizedSum = (short) random().nextInt(0, 4097);
      float normVmC = random().nextFloat(-1000f, 1000f);
      float vDotC = random().nextFloat(-1000f, 1000f);
      float cDotC = VectorUtil.dotProduct(centroids[i], centroids[i]);
      queryVectors[i] =
          new Lucene912BinaryFlatVectorsScorer.BinaryQueryVector(
              vector,
              new BinaryQuantizer.QueryFactors(
                  quantizedSum, distanceToCentroid, vl, width, normVmC, vDotC, cDotC));
    }

    RandomAccessBinarizedByteVectorValues targetVectors =
        new RandomAccessBinarizedByteVectorValues() {
          @Override
          public float getCentroidDistance(int vectorOrd) throws IOException {
            return random().nextFloat(0f, 1000f);
          }

          @Override
          public float getVectorMagnitude(int vectorOrd) throws IOException {
            return random().nextFloat(0f, 100f);
          }

          @Override
          public float getOOQ(int targetOrd) throws IOException {
            return random().nextFloat(-1000f, 1000f);
          }

          @Override
          public float getNormOC(int targetOrd) throws IOException {
            return random().nextFloat(-1000f, 1000f);
          }

          @Override
          public float getODotC(int targetOrd) throws IOException {
            return random().nextFloat(-1000f, 1000f);
          }

          @Override
          public short getClusterId(int vectorOrd) throws IOException {
            return (short) random().nextInt(0, 2);
          }

          @Override
          public BinaryQuantizer getQuantizer() {
            int dimensions = 128;
            return new BinaryQuantizer(dimensions, dimensions, VectorSimilarityFunction.EUCLIDEAN);
          }

          @Override
          public float[][] getCentroids() throws IOException {
            return centroids;
          }

          @Override
          public RandomAccessBinarizedByteVectorValues copy() throws IOException {
            return null;
          }

          @Override
          public byte[] vectorValue(int targetOrd) throws IOException {
            byte[] vectorBytes = new byte[discretizedDimensions / 8];
            random().nextBytes(vectorBytes);
            return vectorBytes;
          }

          @Override
          public int size() {
            return 1;
          }

          @Override
          public int dimension() {
            return dimensions;
          }
        };

    Lucene912BinaryFlatVectorsScorer.BinarizedRandomVectorScorer scorer =
        new Lucene912BinaryFlatVectorsScorer.BinarizedRandomVectorScorer(
            queryVectors, targetVectors, similarityFunction, discretizedDimensions);

    float score = scorer.score(0);

    assertTrue(score >= 0f);
  }

  public void testScoreEuclidean() throws IOException {
    int dimensions = 128;

    Lucene912BinaryFlatVectorsScorer.BinaryQueryVector[] queryVectors =
        new Lucene912BinaryFlatVectorsScorer.BinaryQueryVector[1];
    byte[] vector =
        new byte[] {
          -8, 10, -27, 112, -83, 36, -36, -122, -114, 82, 55, 33, -33, 120, 55, -99, -93, -86, -55,
          21, -121, 30, 111, 30, 0, 82, 21, 38, -120, -127, 40, -32, 78, -37, 42, -43, 122, 115, 30,
          115, 123, 108, -13, -65, 123, 124, -33, -68, 49, 5, 20, 58, 0, 12, 30, 30, 4, 97, 10, 66,
          4, 35, 1, 67
        };
    float distanceToCentroid = 157799.12f;
    float vl = -57.883f;
    float width = 9.972266f;
    short quantizedSum = 795;
    queryVectors[0] =
        new Lucene912BinaryFlatVectorsScorer.BinaryQueryVector(
            vector,
            new BinaryQuantizer.QueryFactors(
                quantizedSum, distanceToCentroid, vl, width, 0f, 0f, 0f));

    RandomAccessBinarizedByteVectorValues targetVectors =
        new RandomAccessBinarizedByteVectorValues() {
          @Override
          public float getCentroidDistance(int vectorOrd) throws IOException {
            return 355.78073f;
          }

          @Override
          public float getVectorMagnitude(int vectorOrd) throws IOException {
            return 0.7636705f;
          }

          @Override
          public float getOOQ(int targetOrd) throws IOException {
            return 0;
          }

          @Override
          public float getNormOC(int targetOrd) throws IOException {
            return 0;
          }

          @Override
          public float getODotC(int targetOrd) throws IOException {
            return 0;
          }

          @Override
          public short getClusterId(int vectorOrd) throws IOException {
            return 0;
          }

          @Override
          public BinaryQuantizer getQuantizer() {
            int dimensions = 128;
            return new BinaryQuantizer(dimensions, dimensions, VectorSimilarityFunction.EUCLIDEAN);
          }

          @Override
          public float[][] getCentroids() throws IOException {
            return new float[][] {
              {
                26.7f, 16.2f, 10.913f, 10.314f, 12.12f, 14.045f, 15.887f, 16.864f, 32.232f, 31.567f,
                34.922f, 21.624f, 16.349f, 29.625f, 31.994f, 22.044f, 37.847f, 24.622f, 36.299f,
                27.966f, 14.368f, 19.248f, 30.778f, 35.927f, 27.019f, 16.381f, 17.325f, 16.517f,
                13.272f, 9.154f, 9.242f, 17.995f, 53.777f, 23.011f, 12.929f, 16.128f, 22.16f,
                28.643f, 25.861f, 27.197f, 59.883f, 40.878f, 34.153f, 22.795f, 24.402f, 37.427f,
                34.19f, 29.288f, 61.812f, 26.355f, 39.071f, 37.789f, 23.33f, 22.299f, 28.64f,
                47.828f, 52.457f, 21.442f, 24.039f, 29.781f, 27.707f, 19.484f, 14.642f, 28.757f,
                54.567f, 20.936f, 25.112f, 25.521f, 22.077f, 18.272f, 14.526f, 29.054f, 61.803f,
                24.509f, 37.517f, 35.906f, 24.106f, 22.64f, 32.1f, 48.788f, 60.102f, 39.625f,
                34.766f, 22.497f, 24.397f, 41.599f, 38.419f, 30.99f, 55.647f, 25.115f, 14.96f,
                18.882f, 26.918f, 32.442f, 26.231f, 27.107f, 26.828f, 15.968f, 18.668f, 14.071f,
                10.906f, 8.989f, 9.721f, 17.294f, 36.32f, 21.854f, 35.509f, 27.106f, 14.067f,
                19.82f, 33.582f, 35.997f, 33.528f, 30.369f, 36.955f, 21.23f, 15.2f, 30.252f, 34.56f,
                22.295f, 29.413f, 16.576f, 11.226f, 10.754f, 12.936f, 15.525f, 15.868f, 16.43f
              }
            };
          }

          @Override
          public RandomAccessBinarizedByteVectorValues copy() throws IOException {
            return null;
          }

          @Override
          public byte[] vectorValue(int targetOrd) throws IOException {
            return new byte[] {
              44, 108, 120, -15, -61, -32, 124, 25, -63, -57, 6, 24, 1, -61, 1, 14
            };
          }

          @Override
          public int size() {
            return 1;
          }

          @Override
          public int dimension() {
            return dimensions;
          }
        };

    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

    int discretizedDimensions = dimensions;

    Lucene912BinaryFlatVectorsScorer.BinarizedRandomVectorScorer scorer =
        new Lucene912BinaryFlatVectorsScorer.BinarizedRandomVectorScorer(
            queryVectors, targetVectors, similarityFunction, discretizedDimensions);

    assertEquals(1f / (1f + 245482.47f), scorer.score(0), 0.1f);
  }

  public void testScoreMIP() throws IOException {
    int dimensions = 768;

    Lucene912BinaryFlatVectorsScorer.BinaryQueryVector[] queryVectors =
        new Lucene912BinaryFlatVectorsScorer.BinaryQueryVector[1];
    byte[] vector =
        new byte[] {
          -76, 44, 81, 31, 30, -59, 56, -118, -36, 45, -11, 8, -61, 95, -100, 18, -91, -98, -46, 31,
          -8, 82, -42, 121, 75, -61, 125, -21, -82, 16, 21, 40, -1, 12, -92, -22, -49, -92, -19,
          -32, -56, -34, 60, -100, 69, 13, 60, -51, 90, 4, -77, 63, 124, 69, 88, 73, -72, 29, -96,
          44, 69, -123, -59, -94, 84, 80, -61, 27, -37, -92, -51, -86, 19, -55, -36, -2, 68, -37,
          -128, 59, -47, 119, -53, 56, -12, 37, 27, 119, -37, 125, 78, 19, 15, -9, 94, 100, -72, 55,
          86, -48, 26, 10, -112, 28, -15, -64, -34, 55, -42, -31, -96, -18, 60, -44, 69, 106, -20,
          15, 47, 49, -122, -45, 119, 101, 22, 77, 108, -15, -71, -28, -43, -68, -127, -86, -118,
          -51, 121, -65, -10, -49, 115, -6, -61, -98, 21, 41, 56, 29, -16, -82, 4, 72, -77, 23, 23,
          -32, -98, 112, 27, -4, 91, -69, 102, -114, 16, -20, -76, -124, 43, 12, 3, -30, 42, -44,
          -88, -72, -76, -94, -73, 46, -17, 4, -74, -44, 53, -11, -117, -105, -113, -37, -43, -128,
          -70, 56, -68, -100, 56, -20, 77, 12, 17, -119, -17, 59, -10, -26, 29, 42, -59, -28, -28,
          60, -34, 60, -24, 80, -81, 24, 122, 127, 62, 124, -5, -11, 59, -52, 74, -29, -116, 3, -40,
          -99, -24, 11, -10, 95, 21, -38, 59, -52, 29, 58, 112, 100, -106, -90, 71, 72, 57, 95, 98,
          96, -41, -16, 50, -18, 123, -36, 74, -101, 17, 50, 48, 96, 57, 7, 81, -16, -32, -102, -24,
          -71, -10, 37, -22, 94, -36, -52, -71, -47, 47, -1, -31, -10, -126, -15, -123, -59, 71,
          -49, 67, 99, -57, 21, -93, -13, -18, 54, -112, -60, 9, 25, -30, -47, 26, 27, 26, -63, 1,
          -63, 18, -114, 80, 110, -123, 0, -63, -126, -128, 10, -60, 51, -71, 28, 114, -4, 53, 10,
          23, -96, 9, 32, -22, 5, -108, 33, 98, -59, -106, -126, 73, 72, -72, -73, -60, -96, -99,
          31, 40, 15, -19, 17, -128, 33, -75, 96, -18, -47, 75, 27, -60, -16, -82, 13, 21, 37, 23,
          70, 9, -39, 16, -127, 35, -78, 64, 99, -46, 1, 28, 65, 125, 14, 42, 26
        };
    float distanceToCentroid = 95.39032f;
    float vl = -0.10079563f;
    float width = 0.014609014f;
    short quantizedSum = 5306;
    float normVmC = 9.766797f;
    float vDotC = 133.56123f;
    float cDotC = 132.20227f;
    queryVectors[0] =
        new Lucene912BinaryFlatVectorsScorer.BinaryQueryVector(
            vector,
            new BinaryQuantizer.QueryFactors(
                quantizedSum, distanceToCentroid, vl, width, normVmC, vDotC, cDotC));

    RandomAccessBinarizedByteVectorValues targetVectors =
        new RandomAccessBinarizedByteVectorValues() {
          @Override
          public float getCentroidDistance(int vectorOrd) throws IOException {
            return 0f;
          }

          @Override
          public float getVectorMagnitude(int vectorOrd) throws IOException {
            return 0f;
          }

          @Override
          public float getOOQ(int targetOrd) throws IOException {
            return 0.7882396f;
          }

          @Override
          public float getNormOC(int targetOrd) throws IOException {
            return 5.0889387f;
          }

          @Override
          public float getODotC(int targetOrd) throws IOException {
            return 131.485660f;
          }

          @Override
          public short getClusterId(int vectorOrd) throws IOException {
            return 0;
          }

          @Override
          public BinaryQuantizer getQuantizer() {
            int dimensions = 768;
            return new BinaryQuantizer(
                dimensions, dimensions, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
          }

          @Override
          public float[][] getCentroids() throws IOException {
            return new float[][] {
              {
                0.16672021f, 0.11700719f, 0.013227397f, 0.09305186f, -0.029422699f, 0.17622353f,
                0.4267106f, -0.297038f, 0.13915674f, 0.38441318f, -0.486725f, -0.15987667f,
                -0.19712289f, 0.1349074f, -0.19016947f, -0.026179956f, 0.4129807f, 0.14325741f,
                -0.09106042f, 0.06876218f, -0.19389102f, 0.4467732f, 0.03169017f, -0.066950575f,
                -0.044301506f, -0.0059755715f, -0.33196586f, 0.18213534f, -0.25065416f, 0.30251458f,
                0.3448419f, -0.14900115f, -0.07782894f, 0.3568707f, -0.46595258f, 0.37295088f,
                -0.088741764f, 0.17248306f, -0.0072736046f, 0.32928637f, 0.13216197f, 0.032092985f,
                0.21553043f, 0.016091486f, 0.31958902f, 0.0133126f, 0.1579258f, 0.018537233f,
                0.046248164f, -0.0048194043f, -0.2184672f, -0.26273906f, -0.110678785f,
                    -0.04542999f,
                -0.41625032f, 0.46025568f, -0.16116948f, 0.4091706f, 0.18427321f, 0.004736977f,
                0.16289745f, -0.05330932f, -0.2694863f, -0.14762327f, 0.17744702f, 0.2445075f,
                0.14377175f, 0.37390858f, 0.16165806f, 0.17177118f, 0.097307935f, 0.36326465f,
                0.23221572f, 0.15579978f, -0.065486655f, -0.29006517f, -0.009194494f, 0.009019374f,
                0.32154799f, -0.23186184f, 0.46485493f, -0.110756285f, -0.18604982f, 0.35027295f,
                0.19815539f, 0.47386464f, -0.031379268f, 0.124035835f, 0.11556784f, 0.4304302f,
                -0.24455063f, 0.1816723f, 0.034300473f, -0.034347706f, 0.040140998f, 0.1389901f,
                0.22840638f, -0.19911191f, 0.07563166f, -0.2744902f, 0.13114859f, -0.23862572f,
                -0.31404558f, 0.41355187f, 0.12970817f, -0.35403475f, -0.2714075f, 0.07231573f,
                0.043893218f, 0.30324167f, 0.38928393f, -0.1567055f, -0.0083288215f, 0.0487653f,
                0.12073729f, -0.01582117f, 0.13381198f, -0.084824145f, -0.15329859f, -1.120622f,
                0.3972598f, 0.36022213f, -0.29826534f, -0.09468781f, 0.03550699f, -0.21630692f,
                0.55655843f, -0.14842057f, 0.5924833f, 0.38791573f, 0.1502777f, 0.111737385f,
                0.1926823f, 0.66021144f, 0.25601995f, 0.28220543f, 0.10194068f, 0.013066262f,
                -0.09348819f, -0.24085014f, -0.17843121f, -0.012598432f, 0.18757571f, 0.48543528f,
                -0.059388146f, 0.1548026f, 0.041945867f, 0.3322589f, 0.012830887f, 0.16621992f,
                0.22606649f, 0.13959105f, -0.16688728f, 0.47194278f, -0.12767595f, 0.037815034f,
                0.441938f, 0.07875027f, 0.08625042f, 0.053454693f, 0.74093896f, 0.34662113f,
                0.009829135f, -0.033400282f, 0.030965377f, 0.17645596f, 0.083803624f, 0.32578796f,
                0.49538168f, -0.13212465f, -0.39596975f, 0.109529115f, 0.2815771f, -0.051440604f,
                0.21889819f, 0.25598505f, 0.012208843f, -0.012405662f, 0.3248759f, 0.00997502f,
                0.05999008f, 0.03562817f, 0.19007418f, 0.24805716f, 0.5926766f, 0.26937613f,
                0.25856f, -0.05798439f, -0.29168302f, 0.14050555f, 0.084851265f, -0.03763504f,
                0.8265359f, -0.23383066f, -0.042164285f, 0.19120507f, -0.12189065f, 0.3864055f,
                -0.19823311f, 0.30280992f, 0.10814344f, -0.164514f, -0.22905481f, 0.13680641f,
                0.4513772f, -0.514546f, -0.061746247f, 0.11598224f, -0.23093395f, -0.09735358f,
                0.02767051f, 0.11594536f, 0.17106244f, 0.21301728f, -0.048222974f, 0.2212131f,
                -0.018857865f, -0.09783516f, 0.42156664f, -0.14032331f, -0.103861615f, 0.4190284f,
                0.068923555f, -0.015083771f, 0.083590426f, -0.15759592f, -0.19096768f, -0.4275228f,
                0.12626286f, 0.12192557f, 0.4157616f, 0.048780657f, 0.008426048f, -0.0869124f,
                0.054927208f, 0.28417027f, 0.29765493f, 0.09203619f, -0.14446871f, -0.117514975f,
                0.30662632f, 0.24904715f, -0.19551662f, -0.0045785015f, 0.4217626f, -0.31457824f,
                0.23381722f, 0.089111514f, -0.27170828f, -0.06662652f, 0.10011391f, -0.090274535f,
                0.101849966f, 0.26554734f, -0.1722843f, 0.23296228f, 0.25112453f, -0.16790418f,
                0.010348314f, 0.05061285f, 0.38003662f, 0.0804625f, 0.3450673f, 0.364368f,
                -0.2529952f, -0.034065288f, 0.22796603f, 0.5457553f, 0.11120353f, 0.24596325f,
                0.42822433f, -0.19215727f, -0.06974534f, 0.19388479f, -0.17598474f, -0.08769705f,
                0.12769659f, 0.1371616f, -0.4636819f, 0.16870509f, 0.14217548f, 0.04412187f,
                -0.20930687f, 0.0075530168f, 0.10065227f, 0.45334083f, -0.1097471f, -0.11139921f,
                -0.31835595f, -0.057386875f, 0.16285825f, 0.5088513f, -0.06318843f, -0.34759882f,
                0.21132466f, 0.33609292f, 0.04858872f, -0.058759f, 0.22845529f, -0.07641319f,
                0.5452827f, -0.5050389f, 0.1788054f, 0.37428045f, 0.066334985f, -0.28162515f,
                -0.15629752f, 0.33783385f, -0.0832242f, 0.29144394f, 0.47892854f, -0.47006592f,
                -0.07867588f, 0.3872869f, 0.28053126f, 0.52399015f, 0.21979983f, 0.076880336f,
                0.47866163f, 0.252952f, -0.1323851f, -0.22225754f, -0.38585815f, 0.12967427f,
                0.20340872f, -0.326928f, 0.09636557f, -0.35929212f, 0.5413311f, 0.019960884f,
                0.33512768f, 0.15133342f, -0.14124066f, -0.1868793f, -0.07862198f, 0.22739467f,
                0.19598985f, 0.34314656f, -0.05071516f, -0.21107961f, 0.19934991f, 0.04822684f,
                0.15060754f, 0.26586458f, -0.15528078f, 0.123646654f, 0.14450715f, -0.12574252f,
                0.30608323f, 0.018549249f, 0.36323825f, 0.06762097f, 0.08562406f, -0.07863075f,
                0.15975896f, 0.008347004f, 0.37931192f, 0.22957338f, 0.33606857f, -0.25204057f,
                0.18126069f, 0.41903302f, 0.20244692f, -0.053850617f, 0.23088565f, 0.16085246f,
                0.1077502f, -0.12445943f, 0.115779735f, 0.124704875f, 0.13076028f, -0.11628619f,
                -0.12580182f, 0.065204754f, -0.26290357f, -0.23539798f, -0.1855292f, 0.39872098f,
                0.44495568f, 0.05491784f, 0.05135692f, 0.624011f, 0.22839564f, 0.0022447354f,
                -0.27169296f, -0.1694988f, -0.19106841f, 0.0110123325f, 0.15464798f, -0.16269256f,
                0.04033836f, -0.11792753f, 0.17172396f, -0.08912173f, -0.30929542f, -0.03446989f,
                -0.21738084f, 0.39657044f, 0.33550346f, -0.06839139f, 0.053675443f, 0.33783767f,
                0.22576828f, 0.38280004f, 4.1448855f, 0.14225426f, 0.24038498f, 0.072373435f,
                -0.09465926f, -0.016144043f, 0.40864578f, -0.2583055f, 0.031816103f, 0.062555805f,
                0.06068663f, 0.25858644f, -0.10598804f, 0.18201788f, -0.00090025424f, 0.085680895f,
                0.4304161f, 0.028686283f, 0.027298616f, 0.27473378f, -0.3888415f, 0.44825438f,
                0.3600378f, 0.038944595f, 0.49292335f, 0.18556066f, 0.15779617f, 0.29989767f,
                0.39233804f, 0.39759228f, 0.3850708f, -0.0526475f, 0.18572918f, 0.09667526f,
                -0.36111078f, 0.3439669f, 0.1724522f, 0.14074509f, 0.26097745f, 0.16626832f,
                -0.3062964f, -0.054877423f, 0.21702516f, 0.4736452f, 0.2298038f, -0.2983771f,
                0.118479654f, 0.35940516f, 0.12212727f, 0.17234904f, 0.30632678f, 0.09207966f,
                -0.14084268f, -0.19737118f, 0.12442629f, 0.52454203f, 0.1266684f, 0.3062802f,
                0.121598125f, -0.09156268f, 0.11491686f, -0.105715364f, 0.19831072f, 0.061421417f,
                -0.41778997f, 0.14488487f, 0.023310646f, 0.27257463f, 0.16821945f, -0.16702746f,
                0.263203f, 0.33512688f, 0.35117313f, -0.31740817f, -0.14203706f, 0.061256267f,
                -0.19764185f, 0.04822579f, -0.0016218472f, -0.025792575f, 0.4885193f, -0.16942391f,
                -0.04156327f, 0.15908112f, -0.06998626f, 0.53907114f, 0.10317832f, -0.365468f,
                0.4729886f, 0.14291425f, 0.32812154f, -0.0273262f, 0.31760117f, 0.16925456f,
                0.21820979f, 0.085142255f, 0.16118735f, -3.7089362f, 0.251577f, 0.18394576f,
                0.027926167f, 0.15720351f, 0.13084261f, 0.16240814f, 0.23045056f, -0.3966458f,
                0.22822891f, -0.061541352f, 0.028320132f, -0.14736478f, 0.184569f, 0.084853746f,
                0.15172474f, 0.08277542f, 0.27751622f, 0.23450488f, -0.15349835f, 0.29665688f,
                0.32045734f, 0.20012043f, -0.2749372f, 0.011832386f, 0.05976605f, 0.018300122f,
                -0.07855043f, -0.075900674f, 0.0384252f, -0.15101928f, 0.10922137f, 0.47396383f,
                -0.1771141f, 0.2203417f, 0.33174303f, 0.36640546f, 0.10906258f, 0.13765177f,
                0.2488032f, -0.061588854f, 0.20347528f, 0.2574979f, 0.22369152f, 0.18777567f,
                -0.0772263f, -0.1353299f, 0.087077625f, -0.05409276f, 0.027534787f, 0.08053508f,
                0.3403908f, -0.15362988f, 0.07499862f, 0.54367846f, -0.045938436f, 0.12206868f,
                0.031069376f, 0.2972343f, 0.3235321f, -0.053970363f, -0.0042564687f, 0.21447177f,
                0.023565233f, -0.1286087f, -0.047359955f, 0.23021339f, 0.059837278f, 0.19709614f,
                -0.17340347f, 0.11572943f, 0.21720429f, 0.29375625f, -0.045433592f, 0.033339307f,
                0.24594454f, -0.021661613f, -0.12823369f, 0.41809165f, 0.093840264f, -0.007481906f,
                0.22441079f, -0.45719734f, 0.2292629f, 2.675806f, 0.3690025f, 2.1311781f,
                0.07818368f, -0.17055893f, 0.3162922f, -0.2983149f, 0.21211359f, 0.037087034f,
                0.021580033f, 0.086415835f, 0.13541797f, -0.12453424f, 0.04563163f, -0.082379065f,
                -0.15938349f, 0.38595748f, -0.8796574f, -0.080991246f, 0.078572094f, 0.20274459f,
                0.009252143f, -0.12719384f, 0.105845824f, 0.1592398f, -0.08656061f, -0.053054806f,
                0.090986334f, -0.02223379f, -0.18215932f, -0.018316114f, 0.1806707f, 0.24788831f,
                -0.041049056f, 0.01839475f, 0.19160001f, -0.04827654f, 4.4070687f, 0.12640671f,
                -0.11171499f, -0.015480781f, 0.14313947f, 0.10024215f, 0.4129662f, 0.038836367f,
                -0.030228542f, 0.2948598f, 0.32946473f, 0.2237934f, 0.14260699f, -0.044821896f,
                0.23791742f, 0.079720296f, 0.27059034f, 0.32129505f, 0.2725177f, 0.06883333f,
                0.1478041f, 0.07598411f, 0.27230525f, -0.04704308f, 0.045167264f, 0.215413f,
                0.20359069f, -0.092178136f, -0.09523752f, 0.21427691f, 0.10512272f, 5.1295033f,
                0.040909242f, 0.007160441f, -0.192866f, -0.102640584f, 0.21103396f, -0.006780398f,
                -0.049653083f, -0.29426834f, -0.0038102255f, -0.13842082f, 0.06620181f, -0.3196518f,
                0.33279592f, 0.13845938f, 0.16162738f, -0.24798508f, -0.06672485f, 0.195944f,
                -0.11957207f, 0.44237947f, -0.07617347f, 0.13575341f, -0.35074243f, -0.093798876f,
                0.072853446f, -0.20490398f, 0.26504788f, -0.046076056f, 0.16488416f, 0.36007464f,
                0.20955376f, -0.3082038f, 0.46533757f, -0.27326992f, -0.14167665f, 0.25017953f,
                0.062622115f, 0.14057694f, -0.102370486f, 0.33898357f, 0.36456722f, -0.10120469f,
                -0.27838466f, -0.11779602f, 0.18517569f, -0.05942488f, 0.076405466f, 0.007960496f,
                0.0443746f, 0.098998964f, -0.01897129f, 0.8059487f, 0.06991939f, 0.26562217f,
                0.26942885f, 0.11432197f, -0.0055776504f, 0.054493718f, -0.13086213f, 0.6841702f,
                0.121975765f, 0.02787146f, 0.29039973f, 0.30943078f, 0.21762547f, 0.28751117f,
                0.027524523f, 0.5315654f, -0.22451901f, -0.13782433f, 0.08228316f, 0.07808882f,
                0.17445615f, -0.042489477f, 0.13232234f, 0.2756272f, -0.18824948f, 0.14326479f,
                -0.119312495f, 0.011788091f, -0.22103515f, -0.2477118f, -0.10513839f, 0.034028634f,
                0.10693818f, 0.03057979f, 0.04634646f, 0.2289361f, 0.09981585f, 0.26901972f,
                0.1561221f, -0.10639886f, 0.36466748f, 0.06350991f, 0.027927283f, 0.11919768f,
                0.23290513f, -0.03417105f, 0.16698854f, -0.19243467f, 0.28430334f, 0.03754995f,
                -0.08697018f, 0.20413163f, -0.27218238f, 0.13707504f, -0.082289375f, 0.03479585f,
                0.2298305f, 0.4983682f, 0.34522808f, -0.05711886f, -0.10568684f, -0.07771385f
              }
            };
          }

          @Override
          public RandomAccessBinarizedByteVectorValues copy() throws IOException {
            return null;
          }

          @Override
          public byte[] vectorValue(int targetOrd) throws IOException {
            return new byte[] {
              -88, -3, 60, -75, -38, 79, 84, -53, -116, -126, 19, -19, -21, -80, 69, 101, -71, 53,
              101, -124, -24, -76, 92, -45, 108, -107, -18, 102, 23, -80, -47, 116, 87, -50, 27,
              -31, -10, -13, 117, -88, -27, -93, -98, -39, 30, -109, -114, 5, -15, 98, -82, 81, 83,
              118, 30, -118, -12, -95, 121, 125, -13, -88, 75, -85, -56, -126, 82, -59, 48, -81, 67,
              -63, 81, 24, -83, 95, -44, 103, 3, -40, -13, -41, -29, -60, 1, 65, -4, -110, -40, 34,
              118, 51, -76, 75, 70, -51
            };
          }

          @Override
          public int size() {
            return 1;
          }

          @Override
          public int dimension() {
            return dimensions;
          }
        };

    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

    int discretizedDimensions = dimensions;

    Lucene912BinaryFlatVectorsScorer.BinarizedRandomVectorScorer scorer =
        new Lucene912BinaryFlatVectorsScorer.BinarizedRandomVectorScorer(
            queryVectors, targetVectors, similarityFunction, discretizedDimensions);

    assertEquals(132.30249f, scorer.score(0), 0.0001f);
  }
}
