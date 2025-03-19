/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.internal.vectorization;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;

public class TestOSQVectorsScorer extends BaseVectorizationTestCase {

  public void testInt4BitDotProduct() throws Exception {
    final int length = random().nextInt(1, 2000) * 2;
    final int numVectors = random().nextInt(1, 100);
    final byte[][] vectors = new byte[numVectors][length];
    final byte[] query = new byte[4 * length];
    try (Directory dir = new MMapDirectory(createTempDir())) {
      try (IndexOutput out = dir.createOutput("tests.bin", IOContext.DEFAULT)) {
        for (int i = 0; i < numVectors; i++) {
          random().nextBytes(vectors[i]);
          out.writeBytes(vectors[i], 0, length);
        }
      }
      random().nextBytes(query);
      try (IndexInput in = dir.openInput("tests.bin", IOContext.DEFAULT)) {
        // Work on a slice that has just the right number of bytes to make the test fail with an
        // index-out-of-bounds in case the implementation reads more than the allowed number of
        // padding bytes.
        final IndexInput slice = in.slice("test", 0, (long) length * numVectors);
        final OSQVectorsScorer defaultScorer = LUCENE_PROVIDER.newOSQVectorsScorer(slice, length);
        final OSQVectorsScorer panamaScorer = PANAMA_PROVIDER.newOSQVectorsScorer(in, length);
        for (int i = 0; i < numVectors; i++) {
          assertEquals(
              VectorUtil.int4BitDotProduct(query, vectors[i]),
              defaultScorer.int4BitDotProduct(query));
          assertEquals(
              VectorUtil.int4BitDotProduct(query, vectors[i]),
              panamaScorer.int4BitDotProduct(query));
          assertEquals(in.getFilePointer(), slice.getFilePointer());
        }
        assertEquals((long) length * numVectors, slice.getFilePointer());
      }
    }
  }
}
