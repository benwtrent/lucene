/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.GroupVIntUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import javax.sound.midi.SysexMessage;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class IVFUtils {

  static final int BLOCK_SIZE = 64;

  static final Map<Integer, RandomOrtho> cache = new ConcurrentHashMap<>();

  static RandomOrtho forDims(int dim) {
    return cache.computeIfAbsent(dim, IVFUtils::randomOrthogonal);
  }

  static void modifiedGramSchmidt(int dim, float[] m) {
    for (int i = 0; i < dim; ++i) {
      float norm = VectorUtil.dotProduct(dim, m, i * dim, m, i * dim);
      norm = (float) Math.sqrt(norm);
      if (norm == 0f) {
        continue;
      }
      for (int j = 0; j < dim; ++j) {
        m[i * dim + j] /= norm;
      }
      for (int k = i + 1; k < dim; ++k) {
        float dotik = VectorUtil.dotProduct(dim, m, i * dim, m, k * dim);
        for (int j = 0; j < dim; ++j) {
          m[k * dim + j] -= dotik * m[i * dim + j];
        }
      }
    }
  }

  record RandomOrtho(float[][] matrix, int[] dimBlocks) { }

  static void randomFill(Random gen, float[] m) {
    for (int i = 0; i < m.length; ++i) {
      m[i] = (float) gen.nextGaussian();
    }
  }

  private static RandomOrtho randomOrthogonal(int dim) {
    int blockDim = BLOCK_SIZE;
    int nblocks = dim / blockDim;
    int rem = dim % blockDim;

    float[][] blocks = new float[nblocks + (rem > 0 ? 1 : 0)][];
    int[] dimBlocks = new int[nblocks + (rem > 0 ? 1 : 0)];

    Random gen = new Random(215873873);
    float[] m = new float[blockDim * blockDim];
    for (int i = 0; i < nblocks; ++i) {
      randomFill(gen, m);
      modifiedGramSchmidt(blockDim, m);
      blocks[i] = new float[m.length];
      System.arraycopy(m, 0, blocks[i], 0, m.length);
      dimBlocks[i] = blockDim;
    }
    if (rem == 0) {
      return new RandomOrtho(blocks, dimBlocks);
    }

    m = new float[rem * rem];
    randomFill(gen, m);
    modifiedGramSchmidt(rem, m);
    blocks[nblocks] = new float[m.length];
    System.arraycopy(m, 0, blocks[nblocks], 0, m.length);
    dimBlocks[nblocks] = rem;

    return new RandomOrtho(blocks, dimBlocks);
  }

  static class OnlineMeanAndVariance {
    private float mean;
    private float m2;
    private int n;

    public void add(float x) {
      n++;
      float delta = x - mean;
      mean += delta / n;
      m2 += delta * (x - mean);
    }

    public float variance() {
      return m2 / (n - 1f);
    }
  }

  static int minElement(float[] array) {
    int minIndex = 0;
    float minValue = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] < minValue) {
        minValue = array[i];
        minIndex = i;
      }
    }
    return minIndex;
  }

  static int[][] varianceBlockAssignments(FloatVectorValues vectors, RandomOrtho blocks)
    throws IOException {
    OnlineMeanAndVariance[] moments = new OnlineMeanAndVariance[vectors.dimension()];
    final KnnVectorValues.DocIndexIterator iterator = vectors.iterator();
    for (int docV = iterator.nextDoc(); docV != NO_MORE_DOCS; docV = iterator.nextDoc()) {
      float[] vector = vectors.vectorValue(iterator.index());
      for (int j = 0; j < vector.length; ++j) {
        if (moments[j] == null) {
          moments[j] = new OnlineMeanAndVariance();
        }
        moments[j].add(vector[j]);
      }
    }
    Arrays.sort(moments, (a, b) -> Float.compare(b.variance(), a.variance()));
    // create a new array for the block variances
    float[] blockVariances = new float[blocks.matrix.length];
    IntArrayList[] assignment = new IntArrayList[blocks.matrix.length];
    for (int i = 0; i < blocks.matrix.length; ++i) {
      assignment[i] = new IntArrayList();
      int j = minElement(blockVariances);
      assignment[j].add(i);
      blockVariances[j] =
        (assignment[j].size() == blocks.dimBlocks[j]
          ? Float.MAX_VALUE
          : // Prevent further assignments.
          blockVariances[j] + moments[i].variance());
    }
    // convert the IntArrayList to an int[][]
    int[][] finalAssignments = new int[assignment.length][];
    for (int i = 0; i < assignment.length; ++i) {
      finalAssignments[i] = assignment[i].toArray();
    }
    return finalAssignments;
  }

  static void matrixVectorMultiply(int dim, float[] m, float[] x, float[] y) {
    for (int i = 0; i < dim; ++i) {
      y[i] = VectorUtil.dotProduct(dim, m, i * dim, x, 0);
    }
  }

  static void randomOrthogonalTransform(
    float[] vector, RandomOrtho blocks, int[][] assignment, float[] dest) throws IOException {
    float[] v = new float[vector.length];
    float[] x = new float[blocks.dimBlocks[0]];
    float[] y = new float[blocks.dimBlocks[0]];
    System.arraycopy(vector, 0, v, 0, vector.length);
    int i = 0;
    for (int j = 0; j < blocks.matrix.length; ++j) {
      float[] block = blocks.matrix[j];
      int dim_ = blocks.dimBlocks[j];
      for (int k : assignment[j]) {
        x[k] = v[k];
      }
      matrixVectorMultiply(dim_, block, x, y);
      System.arraycopy(y, 0, dest, i, dim_);
      i += dim_;
    }
  }


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
