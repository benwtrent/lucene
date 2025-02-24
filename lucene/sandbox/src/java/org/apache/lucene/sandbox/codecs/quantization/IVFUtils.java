/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.search.DocIdSetIterator;

public class IVFUtils {

  public interface CentroidQueryScorer {
    int size();

    float[] centroid(int centroidOrdinal) throws IOException;

    float score(int centroidOrdinal) throws IOException;
  }

  public interface CentroidAssignmentScorer {
    int size();

    float[] centroid(int centroidOrdinal) throws IOException;

    // TODO maybe we can make this just two ordinals...
    float score(int centroidOrdinal, float[] vector) throws IOException;
  }

  public interface PostingsScorer {
    // TODO maybe we can not specifically pass the centroid...
    DocIdSetIterator resetPostingsScorer(int centroidOrdinal, float[] centroid) throws IOException;

    float score() throws IOException;
  }

  public interface CloseableCentroidAssignmentScorer
      extends CentroidAssignmentScorer, AutoCloseable {
    @Override
    void close();
  }

  public interface VectorCentroidScorer {
    float score(float[] queryVector, int centroidOrdinal);
  }

  static IVFVectorsReader getIVFReader(KnnVectorsReader vectorsReader, String fieldName) {
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
      vectorsReader = candidateReader.getFieldReader(fieldName);
    }
    if (vectorsReader instanceof IVFVectorsReader reader) {
      return reader;
    }
    return null;
  }
}
