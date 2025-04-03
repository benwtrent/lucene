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
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

public class IVFUtils {

  public interface IntIterator {
    int pop() throws IOException;

    boolean hasNext() throws IOException;
  }

  public interface CentroidQueryScorer {
    float[] centroid(int centroidOrdinal) throws IOException;

    IntIterator centroidIterator() throws IOException;
  }

  public interface CentroidAssignmentDistanceEstimator {
    NeighborQueue estimateNearestCentroids(int vectorOrd, int k) throws IOException;
  }

  public interface CentroidAssignmentScorer {
    int size();

    float[] centroid(int centroidOrdinal) throws IOException;

    default CentroidAssignmentDistanceEstimator getEstimator(FloatVectorValues vectorValues)
        throws IOException {
      return null;
    }

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

  static IVFVectorsReader getIVFReader(KnnVectorsReader vectorsReader, String fieldName) {
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
      vectorsReader = candidateReader.getFieldReader(fieldName);
    }
    if (vectorsReader instanceof IVFVectorsReader reader) {
      return reader;
    }
    return null;
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

  static void writeQuantizedValue(
      IndexOutput indexOutput,
      byte[] binaryValue,
      OptimizedScalarQuantizer.QuantizationResult corrections)
      throws IOException {
    indexOutput.writeBytes(binaryValue, binaryValue.length);
    indexOutput.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
    indexOutput.writeInt(Float.floatToIntBits(corrections.upperInterval()));
    indexOutput.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
    assert corrections.quantizedComponentSum() >= 0
        && corrections.quantizedComponentSum() <= 0xffff;
    indexOutput.writeShort((short) corrections.quantizedComponentSum());
  }

  static int readQuantizedValue(IndexInput indexInput, byte[] binaryValue, float[] corrections)
      throws IOException {
    assert corrections.length == 3;
    indexInput.readBytes(binaryValue, 0, binaryValue.length);
    corrections[0] = Float.intBitsToFloat(indexInput.readInt());
    corrections[1] = Float.intBitsToFloat(indexInput.readInt());
    corrections[2] = Float.intBitsToFloat(indexInput.readInt());
    return Short.toUnsignedInt(indexInput.readShort());
  }
}
