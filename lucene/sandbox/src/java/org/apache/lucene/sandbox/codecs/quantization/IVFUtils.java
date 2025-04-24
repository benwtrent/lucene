/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import java.io.IOException;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

public class IVFUtils {

  public interface CentroidQueryScorer {
    int size();

    float[] centroid(int centroidOrdinal) throws IOException;

    float score(int centroidOrdinal) throws IOException;
  }

  public interface CentroidAssignmentScorer {
    int size();

    float[] centroid(int centroidOrdinal) throws IOException;

    void setScoringVector(float[] vector);

    float score(int centroidOrdinal) throws IOException;
  }

  public abstract static class PostingVisitor {
    // TODO maybe we can not specifically pass the centroid...

    /** returns the number of documents in the posting list */
    public abstract int resetPostingsScorer(int centroidOrdinal, float[] centroid)
        throws IOException;

    /** returns the number of scored documents */
    public abstract int visit(KnnCollector collector) throws IOException;
  }

  public interface VectorCentroidScorer {
    float score(float[] queryVector, int centroidOrdinal);
  }

  static int calculateByteLength(int dimension, byte bits) {
    int vectorBytes =
        switch (bits) {
          case 1 -> (OptimizedScalarQuantizer.discretize(dimension, 64) / 8);
          case 4 -> dimension;
          case 32 -> dimension * Float.BYTES;
          default -> throw new IllegalStateException("Unexpected value: " + bits);
        };
    return vectorBytes + 3 * Float.BYTES + Short.BYTES;
  }
}
