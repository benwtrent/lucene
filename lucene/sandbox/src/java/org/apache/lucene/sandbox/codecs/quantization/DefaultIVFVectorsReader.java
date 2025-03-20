/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.QUERY_BITS;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.transposeHalfByte;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.IntPredicate;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.vectorization.OSQVectorsScorer;
import org.apache.lucene.internal.vectorization.VectorizationProvider;
import org.apache.lucene.sandbox.search.knn.IVFKnnSearchStrategy;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.GroupVIntUtil;
import org.apache.lucene.util.PriorityQueue;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Default implementation of {@link IVFVectorsReader}. It scores the posting lists centroids using
 * brute force and then scores the top ones using the posting list.
 *
 * @lucene.experimental
 */
public class DefaultIVFVectorsReader extends IVFVectorsReader {
  private static final float FOUR_BIT_SCALE = 1f / ((1 << 4) - 1);

  static final VectorizationProvider VECTORIZATION_PROVIDER = VectorizationProvider.getInstance();

  public DefaultIVFVectorsReader(SegmentReadState state, FlatVectorsReader rawVectorsReader)
      throws IOException {
    super(state, rawVectorsReader);
  }

  @Override
  protected IVFUtils.CentroidQueryScorer getCentroidScorer(
      FieldInfo fieldInfo, int numCentroids, IndexInput centroids, float[] targetQuery)
      throws IOException {
    float[] globalCentroid = fields.get(fieldInfo.number).globalCentroid();
    float globalCentroidDp = fields.get(fieldInfo.number).globalCentroidDp();
    OptimizedScalarQuantizer scalarQuantizer =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    byte[] quantized = new byte[targetQuery.length];
    float[] targetScratch = ArrayUtil.copyArray(targetQuery);
    OptimizedScalarQuantizer.QuantizationResult queryParams =
        scalarQuantizer.scalarQuantize(targetScratch, quantized, (byte) 4, globalCentroid);
    return new IVFUtils.CentroidQueryScorer() {
      int currentCentroid = -1;
      private final byte[] quantizedCentroid = new byte[fieldInfo.getVectorDimension()];
      private final float[] centroid = new float[fieldInfo.getVectorDimension()];
      private final float[] centroidCorrectiveValues = new float[3];
      private int quantizedCentroidComponentSum;
      private final long centroidByteSize =
          IVFUtils.calculateByteLength(fieldInfo.getVectorDimension(), (byte) 4);

      @Override
      public int size() {
        return numCentroids;
      }

      @Override
      public float[] centroid(int centroidOrdinal) throws IOException {
        readQuantizedCentroid(centroidOrdinal);
        return centroid;
      }

      private void readQuantizedCentroid(int centroidOrdinal) throws IOException {
        if (centroidOrdinal == currentCentroid) {
          return;
        }
        centroids.seek(centroidOrdinal * centroidByteSize);
        quantizedCentroidComponentSum =
            IVFUtils.readQuantizedValue(centroids, quantizedCentroid, centroidCorrectiveValues);
        centroids.seek(
            numCentroids * centroidByteSize
                + (long) Float.BYTES * quantizedCentroid.length * centroidOrdinal);
        centroids.readFloats(centroid, 0, centroid.length);
        currentCentroid = centroidOrdinal;
      }

      @Override
      public float score(int centroidOrdinal) throws IOException {
        readQuantizedCentroid(centroidOrdinal);
        return int4QuantizedScore(
            quantized,
            queryParams,
            fieldInfo.getVectorDimension(),
            quantizedCentroid,
            centroidCorrectiveValues,
            quantizedCentroidComponentSum,
            globalCentroidDp,
            fieldInfo.getVectorSimilarityFunction());
      }
    };
  }

  @Override
  protected FloatVectorValues getCentroids(
      IndexInput indexInput, int numCentroids, FieldInfo info) {
    FieldEntry entry = fields.get(info.number);
    if (entry == null) {
      return null;
    }
    return new OffHeapCentroidFloatVectorValues(
        numCentroids, indexInput, info.getVectorDimension());
  }

  @Override
  protected Iterator<PostingListWithFileOffsetWithScore> scorePostingLists(
      FieldInfo fieldInfo,
      KnnCollector knnCollector,
      IVFUtils.CentroidQueryScorer centroidQueryScorer,
      int nProbe)
      throws IOException {
    List<Integer> preferredCentroids = null;
    if (knnCollector.getSearchStrategy() instanceof IVFKnnSearchStrategy searchStrategy) {
      preferredCentroids = searchStrategy.getCentroids();
    }
    if (preferredCentroids != null && preferredCentroids.isEmpty()) {
      return Arrays.stream(new PostingListWithFileOffsetWithScore[0]).iterator();
    }
    int centroidsToReturn = nProbe;
    if (centroidsToReturn <= 0) {
      centroidsToReturn = Math.max(((knnCollector.k() * 300) / 1_000), 1);
    }
    if (preferredCentroids != null) {
      centroidsToReturn = Math.min(centroidsToReturn, preferredCentroids.size());
    }
    FieldEntry fieldEntry = fields.get(fieldInfo.number);
    // TODO: improve the heuristic here. It does not work they there are many deleted documents or
    // restrictive filter.
    final int postingListsToScore =
        preferredCentroids == null
            ? Math.min(centroidQueryScorer.size(), centroidsToReturn)
            : centroidsToReturn;
    final PriorityQueue<PostingListWithFileOffsetWithScore> pq =
        new PriorityQueue<>(postingListsToScore) {
          @Override
          protected boolean lessThan(
              PostingListWithFileOffsetWithScore a, PostingListWithFileOffsetWithScore b) {
            return a.score() < b.score();
          }
        };
    if (preferredCentroids != null) {
      for (int centroid : preferredCentroids) {
        knnCollector.incVisitedClusterCount(1);
        float score = centroidQueryScorer.score(centroid);
        pq.insertWithOverflow(
            new PostingListWithFileOffsetWithScore(
                new PostingListWithFileOffset(
                    centroid, fieldEntry.postingListOffsetsAndLengths()[centroid]),
                score));
      }
    } else {
      for (int centroid = 0; centroid < centroidQueryScorer.size(); centroid++) {
        knnCollector.incVisitedClusterCount(1);
        float score = centroidQueryScorer.score(centroid);
        pq.insertWithOverflow(
            new PostingListWithFileOffsetWithScore(
                new PostingListWithFileOffset(
                    centroid, fieldEntry.postingListOffsetsAndLengths()[centroid]),
                score));
      }
    }

    final PostingListWithFileOffsetWithScore[] topCentroids =
        new PostingListWithFileOffsetWithScore[postingListsToScore];
    for (int i = 1; i <= postingListsToScore; i++) {
      topCentroids[postingListsToScore - i] = pq.pop();
    }
    return Arrays.stream(topCentroids).iterator();
  }

  static void prefixSum(int[] buffer, int count) {
    for (int i = 1; i < count; ++i) {
      buffer[i] += buffer[i - 1];
    }
  }

  @Override
  protected IVFUtils.PostingsScorer getPostingScorer(
      FieldInfo fieldInfo, IndexInput indexInput, float[] target, IntPredicate needsScoring) {
    FieldEntry entry = fields.get(fieldInfo.number);
    return new MemorySegmentPostingsScorer(target, indexInput, entry, fieldInfo, needsScoring);
  }

  static float int4QuantizedScore(
      byte[] quantizedQuery,
      OptimizedScalarQuantizer.QuantizationResult queryCorrections,
      int dims,
      byte[] binaryCode,
      float[] targetCorrections,
      int targetComponentSum,
      float centroidDp,
      VectorSimilarityFunction similarityFunction) {
    float qcDist = VectorUtil.int4DotProduct(quantizedQuery, binaryCode);
    float ax = targetCorrections[0];
    // Here we assume `lx` is simply bit vectors, so the scaling isn't necessary
    float lx = (targetCorrections[1] - ax) * FOUR_BIT_SCALE;
    float ay = queryCorrections.lowerInterval();
    float ly = (queryCorrections.upperInterval() - ay) * FOUR_BIT_SCALE;
    float y1 = queryCorrections.quantizedComponentSum();
    float score =
        ax * ay * dims + ay * lx * (float) targetComponentSum + ax * ly * y1 + lx * ly * qcDist;
    if (similarityFunction == EUCLIDEAN) {
      score = queryCorrections.additionalCorrection() + targetCorrections[2] - 2 * score;
      return Math.max(1 / (1f + score), 0);
    } else {
      // For cosine and max inner product, we need to apply the additional correction, which is
      // assumed to be the non-centered dot-product between the vector and the centroid
      score += queryCorrections.additionalCorrection() + targetCorrections[2] - centroidDp;
      if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
        return VectorUtil.scaleMaxInnerProductScore(score);
      }
      return Math.max((1f + score) / 2f, 0);
    }
  }

  static float quantizedScore(
      float qcDist,
      OptimizedScalarQuantizer.QuantizationResult queryCorrections,
      int dims,
      float[] targetCorrections,
      int targetComponentSum,
      float centroidDp,
      VectorSimilarityFunction similarityFunction) {
    float ax = targetCorrections[0];
    // Here we assume `lx` is simply bit vectors, so the scaling isn't necessary
    float lx = targetCorrections[1] - ax;
    float ay = queryCorrections.lowerInterval();
    float ly = (queryCorrections.upperInterval() - ay) * FOUR_BIT_SCALE;
    float y1 = queryCorrections.quantizedComponentSum();
    float score =
        ax * ay * dims + ay * lx * (float) targetComponentSum + ax * ly * y1 + lx * ly * qcDist;
    // For euclidean, we need to invert the score and apply the additional correction, which is
    // assumed to be the squared l2norm of the centroid centered vectors.
    if (similarityFunction == EUCLIDEAN) {
      score = queryCorrections.additionalCorrection() + targetCorrections[2] - 2 * score;
      return Math.max(1 / (1f + score), 0);
    } else {
      // For cosine and max inner product, we need to apply the additional correction, which is
      // assumed to be the non-centered dot-product between the vector and the centroid
      score += queryCorrections.additionalCorrection() + targetCorrections[2] - centroidDp;
      if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
        return VectorUtil.scaleMaxInnerProductScore(score);
      }
      return Math.max((1f + score) / 2f, 0);
    }
  }

  static class OffHeapCentroidFloatVectorValues extends FloatVectorValues {
    private final int numCentroids;
    private final IndexInput input;
    private final int dimension;
    private final float[] centroid;
    private final long centroidByteSize;
    private int ord = -1;

    OffHeapCentroidFloatVectorValues(int numCentroids, IndexInput input, int dimension) {
      this.numCentroids = numCentroids;
      this.input = input;
      this.dimension = dimension;
      this.centroid = new float[dimension];
      this.centroidByteSize = IVFUtils.calculateByteLength(dimension, (byte) 4);
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
      if (ord < 0 || ord >= numCentroids) {
        throw new IllegalArgumentException("ord must be in [0, " + numCentroids + "]");
      }
      if (ord == this.ord) {
        return centroid;
      }
      readQuantizedCentroid(ord);
      return centroid;
    }

    private void readQuantizedCentroid(int centroidOrdinal) throws IOException {
      if (centroidOrdinal == ord) {
        return;
      }
      input.seek(
          numCentroids * centroidByteSize + (long) Float.BYTES * dimension * centroidOrdinal);
      input.readFloats(centroid, 0, centroid.length);
      ord = centroidOrdinal;
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public int size() {
      return numCentroids;
    }

    @Override
    public FloatVectorValues copy() throws IOException {
      return new OffHeapCentroidFloatVectorValues(numCentroids, input.clone(), dimension);
    }
  }

  private static class MemorySegmentPostingsScorer extends IVFUtils.PostingsScorer {
    final long quantizedByteLength;
    final IndexInput indexInput;
    final float[] target;
    final FieldEntry entry;
    final FieldInfo fieldInfo;
    final IntPredicate needsScoring;

    int[] docIdsScratch = new int[0];
    IndexInput postingsSlice;
    int vectors;
    boolean quantized = false;
    float centroidDp;
    float[] centroid;
    int pos;
    long slicePos;
    OptimizedScalarQuantizer.QuantizationResult queryCorrections;

    final float[] scratch;
    final byte[] quantizationScratch;
    final byte[] quantizedQueryScratch;
    final OptimizedScalarQuantizer quantizer;
    final float[] correctiveValues = new float[3];
    private OSQVectorsScorer osqVectorsScorer;

    MemorySegmentPostingsScorer(
        float[] target,
        IndexInput indexInput,
        FieldEntry entry,
        FieldInfo fieldInfo,
        IntPredicate needsScoring) {
      this.target = target;
      this.indexInput = indexInput;
      this.entry = entry;
      this.fieldInfo = fieldInfo;
      this.needsScoring = needsScoring;

      scratch = new float[target.length];
      quantizationScratch = new byte[target.length];
      final int discretizedDimensions = discretize(fieldInfo.getVectorDimension(), 64);
      quantizedQueryScratch = new byte[QUERY_BITS * discretizedDimensions / 8];
      quantizedByteLength = discretizedDimensions / 8 + (Float.BYTES * 3) + Short.BYTES;
      quantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    }

    @Override
    public void resetPostingsScorer(int centroidOrdinal, float[] centroid) throws IOException {
      quantized = false;
      postingsSlice = entry.postingsSlice(indexInput, centroidOrdinal);
      vectors = postingsSlice.readVInt();
      centroidDp = Float.intBitsToFloat(postingsSlice.readInt());
      this.centroid = centroid;
      // read the doc ids
      docIdsScratch =
          vectors > docIdsScratch.length
              ? ArrayUtil.growExact(docIdsScratch, vectors)
              : docIdsScratch;
      GroupVIntUtil.readGroupVInts(postingsSlice, docIdsScratch, vectors);
      prefixSum(docIdsScratch, vectors);
      slicePos = postingsSlice.getFilePointer();
      osqVectorsScorer =
          VECTORIZATION_PROVIDER.newOSQVectorsScorer(
              postingsSlice, quantizedQueryScratch.length / 4);
      pos = -1;
    }

    @Override
    public int docID() {
      return docIdsScratch[pos];
    }

    @Override
    public int nextDoc() {
      while (true) {
        pos++;
        if (pos < vectors) {
          int docID = docID();
          if (needsScoring.test(docID)) {
            return docID;
          }
        } else {
          return NO_MORE_DOCS;
        }
      }
    }

    @Override
    public int advance(int target) throws IOException {
      return slowAdvance(target);
    }

    @Override
    public long cost() {
      return vectors;
    }

    @Override
    public float score() throws IOException {
      if (quantized == false) {
        System.arraycopy(target, 0, scratch, 0, target.length);
        if (fieldInfo.getVectorSimilarityFunction() == COSINE) {
          VectorUtil.l2normalize(scratch);
        }
        queryCorrections =
            quantizer.scalarQuantize(scratch, quantizationScratch, (byte) 4, centroid);
        transposeHalfByte(quantizationScratch, quantizedQueryScratch);
        quantized = true;
      }
      postingsSlice.seek(slicePos + pos * quantizedByteLength);
      final float qcDist = osqVectorsScorer.int4BitDotProduct(quantizedQueryScratch);
      postingsSlice.readFloats(correctiveValues, 0, correctiveValues.length);
      final int quantizedComponentSum = Short.toUnsignedInt(postingsSlice.readShort());
      return quantizedScore(
          qcDist,
          queryCorrections,
          fieldInfo.getVectorDimension(),
          correctiveValues,
          quantizedComponentSum,
          centroidDp,
          fieldInfo.getVectorSimilarityFunction());
    }
  }
}
