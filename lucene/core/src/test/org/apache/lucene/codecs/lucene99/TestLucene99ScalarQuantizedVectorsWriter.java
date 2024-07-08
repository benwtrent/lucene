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

package org.apache.lucene.codecs.lucene99;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomBoolean;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomByte;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.quantization.ScalarQuantizer;

public class TestLucene99ScalarQuantizedVectorsWriter extends LuceneTestCase {

  public void testFieldWriterRamSizeEstimate() throws IOException {
    int vectorDimension = 10;
    FieldInfo fieldInfo =
        new FieldInfo(
            "foo",
            -1,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            false,
            -1,
            new HashMap<>(),
            0,
            0,
            0,
            vectorDimension,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false);
    TestKnnVectorFieldWriter delegate = new TestKnnVectorFieldWriter();
    Lucene99ScalarQuantizedVectorsWriter.FieldWriter fieldWriter =
        new Lucene99ScalarQuantizedVectorsWriter.FieldWriter(
            1f, randomByte(), randomBoolean(), fieldInfo, InfoStream.NO_OUTPUT, delegate);

    assertEquals(
        Lucene99ScalarQuantizedVectorsWriter.FieldWriter.SHALLOW_SIZE, fieldWriter.ramBytesUsed());
    for (int i = 0; i < 10; i++) {
      fieldWriter.addValue(i, new float[vectorDimension]);
    }
    // confidence interval of 1 should have no additional over head in the quantizer other than
    // references to the vectors
    assertEquals(
        delegate.ramBytesUsed
            + Lucene99ScalarQuantizedVectorsWriter.FieldWriter.SHALLOW_SIZE
            + 10L * RamUsageEstimator.NUM_BYTES_OBJECT_REF,
        fieldWriter.ramBytesUsed());

    delegate = new TestKnnVectorFieldWriter();
    fieldWriter =
        new Lucene99ScalarQuantizedVectorsWriter.FieldWriter(
            randomBoolean() ? 0f : 0.99f,
            randomByte(),
            randomBoolean(),
            fieldInfo,
            InfoStream.NO_OUTPUT,
            delegate);

    assertEquals(
        Lucene99ScalarQuantizedVectorsWriter.FieldWriter.SHALLOW_SIZE, fieldWriter.ramBytesUsed());

    for (int i = 0; i < 10; i++) {
      fieldWriter.addValue(i, new float[vectorDimension]);
    }
    // confidence interval of 0.99 or 0 should have additional over head
    assertTrue(
        delegate.ramBytesUsed
                + Lucene99ScalarQuantizedVectorsWriter.FieldWriter.SHALLOW_SIZE
                + 10L * RamUsageEstimator.NUM_BYTES_OBJECT_REF
            < fieldWriter.ramBytesUsed());
  }

  public void testBuildScalarQuantizerCosine() throws IOException {
    assertScalarQuantizer(
        new float[] {0.3234983f, 0.6236096f}, 0.9f, (byte) 7, VectorSimilarityFunction.COSINE);
    assertScalarQuantizer(
        new float[] {0.28759837f, 0.62449116f}, 0f, (byte) 7, VectorSimilarityFunction.COSINE);
    assertScalarQuantizer(
        new float[] {0.3234983f, 0.6236096f}, 0.9f, (byte) 4, VectorSimilarityFunction.COSINE);
    assertScalarQuantizer(
        new float[] {0.37247902f, 0.58848244f}, 0f, (byte) 4, VectorSimilarityFunction.COSINE);
  }

  public void testBuildScalarQuantizerDotProduct() throws IOException {
    assertScalarQuantizer(
        new float[] {0.3234983f, 0.6236096f}, 0.9f, (byte) 7, VectorSimilarityFunction.DOT_PRODUCT);
    assertScalarQuantizer(
        new float[] {0.28759837f, 0.62449116f}, 0f, (byte) 7, VectorSimilarityFunction.DOT_PRODUCT);
    assertScalarQuantizer(
        new float[] {0.3234983f, 0.6236096f}, 0.9f, (byte) 4, VectorSimilarityFunction.DOT_PRODUCT);
    assertScalarQuantizer(
        new float[] {0.37247902f, 0.58848244f}, 0f, (byte) 4, VectorSimilarityFunction.DOT_PRODUCT);
  }

  public void testBuildScalarQuantizerMIP() throws IOException {
    assertScalarQuantizer(
        new float[] {2.0f, 20.0f}, 0.9f, (byte) 7, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    assertScalarQuantizer(
        new float[] {2.4375f, 19.0625f},
        0f,
        (byte) 7,
        VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    assertScalarQuantizer(
        new float[] {2.0f, 20.0f}, 0.9f, (byte) 4, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    assertScalarQuantizer(
        new float[] {2.6875f, 19.0625f},
        0f,
        (byte) 4,
        VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
  }

  public void testBuildScalarQuantizerEuclidean() throws IOException {
    assertScalarQuantizer(
        new float[] {2.0f, 20.0f}, 0.9f, (byte) 7, VectorSimilarityFunction.EUCLIDEAN);
    assertScalarQuantizer(
        new float[] {2.125f, 19.375f}, 0f, (byte) 7, VectorSimilarityFunction.EUCLIDEAN);
    assertScalarQuantizer(
        new float[] {2.0f, 20.0f}, 0.9f, (byte) 4, VectorSimilarityFunction.EUCLIDEAN);
    assertScalarQuantizer(
        new float[] {2.1875f, 19.0625f}, 0f, (byte) 4, VectorSimilarityFunction.EUCLIDEAN);
  }

  private void assertScalarQuantizer(
      float[] expectedQuantiles,
      Float confidenceInterval,
      byte bits,
      VectorSimilarityFunction vectorSimilarityFunction)
      throws IOException {
    List<float[]> vectors = new ArrayList<>(30);
    for (int i = 0; i < 30; i++) {
      float[] vector = new float[] {i, i + 1, i + 2, i + 3};
      vectors.add(vector);
    }
    FloatVectorValues vectorValues =
        new Lucene99ScalarQuantizedVectorsWriter.FloatVectorWrapper(
            vectors,
            vectorSimilarityFunction == VectorSimilarityFunction.COSINE
                || vectorSimilarityFunction == VectorSimilarityFunction.DOT_PRODUCT);
    ScalarQuantizer scalarQuantizer =
        Lucene99ScalarQuantizedVectorsWriter.buildScalarQuantizer(
            vectorValues, 30, vectorSimilarityFunction, confidenceInterval, bits);
    assertEquals(expectedQuantiles[0], scalarQuantizer.getLowerQuantile(), 0.0001f);
    assertEquals(expectedQuantiles[1], scalarQuantizer.getUpperQuantile(), 0.0001f);
  }

  private static final class TestKnnVectorFieldWriter extends KnnFieldVectorsWriter<float[]> {

    long ramBytesUsed = 0;

    @Override
    public void addValue(int docID, float[] vectorValue) throws IOException {
      ramBytesUsed += RamUsageEstimator.sizeOf(vectorValue);
    }

    @Override
    public float[] copyValue(float[] vectorValue) {
      return vectorValue;
    }

    @Override
    public long ramBytesUsed() {
      return ramBytesUsed;
    }
  }
}
