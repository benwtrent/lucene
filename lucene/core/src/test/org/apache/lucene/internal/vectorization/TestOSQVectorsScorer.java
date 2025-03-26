/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.internal.vectorization;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

public class TestOSQVectorsScorer extends BaseVectorizationTestCase {

  public void testQuantizeScore() throws Exception {
    final int dimensions = random().nextInt(1, 2000);
    final int length = OptimizedScalarQuantizer.discretize(dimensions, 64) / 8;
    final int numVectors = random().nextInt(1, 100);
    final byte[] vector = new byte[length];
    try (Directory dir = new MMapDirectory(createTempDir())) {
      try (IndexOutput out = dir.createOutput("tests.bin", IOContext.DEFAULT)) {
        for (int i = 0; i < numVectors; i++) {
          random().nextBytes(vector);
          out.writeBytes(vector, 0, length);
        }
      }
      final byte[] query = new byte[4 * length];
      random().nextBytes(query);
      try (IndexInput in = dir.openInput("tests.bin", IOContext.DEFAULT)) {
        // Work on a slice that has just the right number of bytes to make the test fail with an
        // index-out-of-bounds in case the implementation reads more than the allowed number of
        // padding bytes.
        final IndexInput slice = in.slice("test", 0, (long) length * numVectors);
        final OSQVectorsScorer defaultScorer =
            LUCENE_PROVIDER.newOSQVectorsScorer(slice, dimensions);
        final OSQVectorsScorer panamaScorer = PANAMA_PROVIDER.newOSQVectorsScorer(in, dimensions);
        for (int i = 0; i < numVectors; i++) {
          assertEquals(defaultScorer.quantizeScore(query), panamaScorer.quantizeScore(query));
          assertEquals(in.getFilePointer(), slice.getFilePointer());
        }
        assertEquals((long) length * numVectors, slice.getFilePointer());
      }
    }
  }

  public void testScore() throws Exception {
    final int dimensions = random().nextInt(1, 2000);
    final int length = OptimizedScalarQuantizer.discretize(dimensions, 64) / 8;
    final int numVectors = OSQVectorsScorer.BULK_SIZE * random().nextInt(1, 10);
    final byte[] vector = new byte[length + 14];
    try (Directory dir = new MMapDirectory(createTempDir())) {
      try (IndexOutput out = dir.createOutput("testScore.bin", IOContext.DEFAULT)) {
        for (int i = 0; i < numVectors; i++) {
          random().nextBytes(vector);
          out.writeBytes(vector, 0, length + 14);
        }
      }
      final byte[] query = new byte[4 * length];
      random().nextBytes(query);
      OptimizedScalarQuantizer.QuantizationResult result =
          new OptimizedScalarQuantizer.QuantizationResult(
              random().nextFloat(),
              random().nextFloat(),
              random().nextFloat(),
              Short.toUnsignedInt((short) random().nextInt()));
      final float centroidDp = random().nextFloat();
      final float[] scores1 = new float[OSQVectorsScorer.BULK_SIZE];
      final float[] scores2 = new float[OSQVectorsScorer.BULK_SIZE];
      for (VectorSimilarityFunction similarityFunction : VectorSimilarityFunction.values()) {
        try (IndexInput in = dir.openInput("testScore.bin", IOContext.DEFAULT)) {
          assertEquals(in.length(), numVectors * (length + 14));
          // Work on a slice that has just the right number of bytes to make the test fail with an
          // index-out-of-bounds in case the implementation reads more than the allowed number of
          // padding bytes.
          for (int i = 0; i < numVectors; i += OSQVectorsScorer.BULK_SIZE) {
            final IndexInput slice =
                in.slice(
                    "test", in.getFilePointer(), (long) (length + 14) * OSQVectorsScorer.BULK_SIZE);
            final OSQVectorsScorer defaultScorer =
                LUCENE_PROVIDER.newOSQVectorsScorer(slice, dimensions);
            final OSQVectorsScorer panamaScorer =
                PANAMA_PROVIDER.newOSQVectorsScorer(in, dimensions);
            defaultScorer.scoreBulk(query, result, similarityFunction, centroidDp, scores1);
            panamaScorer.scoreBulk(query, result, similarityFunction, centroidDp, scores2);
            assertArrayEquals(scores1, scores2, 1e-2f);
            assertEquals(OSQVectorsScorer.BULK_SIZE * (length + 14), slice.getFilePointer());
            assertEquals(OSQVectorsScorer.BULK_SIZE * (length + 14), in.getFilePointer());
          }
        }
      }
    }
  }
}
