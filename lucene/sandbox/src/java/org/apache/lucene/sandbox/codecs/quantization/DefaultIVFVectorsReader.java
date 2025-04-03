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
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
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
      FieldInfo fieldInfo,
      int numCentroids,
      IndexInput centroids,
      float[] targetQuery,
      IndexInput clusters)
      throws IOException {
    FieldEntry fieldEntry = fields.get(fieldInfo.number);
    float[] globalCentroid = fieldEntry.globalCentroid();
    float globalCentroidDp = fieldEntry.globalCentroidDp();
    OptimizedScalarQuantizer scalarQuantizer =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    byte[] quantizedScratch = new byte[targetQuery.length];
    float[] targetScratch = ArrayUtil.copyArray(targetQuery);
    OptimizedScalarQuantizer.QuantizationResult queryParams =
        scalarQuantizer.scalarQuantize(targetScratch, quantizedScratch, (byte) 4, globalCentroid);
    byte[] quantized = new byte[discretize(fieldInfo.getVectorDimension(), 64) / 2];
    OptimizedScalarQuantizer.transposeHalfByte(quantizedScratch, quantized);
    return new BulkQuantizedCentroidQueryScorer(
        numCentroids,
        fieldInfo.getVectorSimilarityFunction(),
        centroids,
        globalCentroidDp,
        targetQuery,
        quantized,
        queryParams);
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
  protected IVFUtils.IntIterator scorePostingLists(
      FieldInfo fieldInfo,
      KnnCollector knnCollector,
      IVFUtils.CentroidQueryScorer centroidQueryScorer,
      int nProbe)
      throws IOException {
    if (knnCollector.getSearchStrategy() instanceof IVFKnnSearchStrategy searchStrategy) {
      if (searchStrategy.getCentroids() != null) {
        throw new IllegalArgumentException("preferred centroids not yet supported");
      }
    }
    return centroidQueryScorer.centroidIterator();
  }

  static void prefixSum(int[] buffer, int count) {
    for (int i = 1; i < count; ++i) {
      buffer[i] += buffer[i - 1];
    }
  }

  @Override
  protected IVFUtils.PostingVisitor getPostingVisitor(
      FieldInfo fieldInfo, IndexInput indexInput, float[] target, IntPredicate needsScoring)
      throws IOException {
    FieldEntry entry = fields.get(fieldInfo.number);
    return new MemorySegmentPostingsVisitor(target, indexInput, entry, fieldInfo, needsScoring);
  }

  static class BulkQuantizedCentroidQueryScorer implements IVFUtils.CentroidQueryScorer {
    final int centroidCount;
    final IndexInput centroids;
    final float globalCentroidDp;
    final float[] centroid;
    int currentCentroid = -1;
    final float[] targetQuery;
    final byte[] quantizedTargetQuery;
    final OptimizedScalarQuantizer.QuantizationResult targetQueryCorrections;
    final int bulkCentroidLoopBound;
    final int centroidByteSize;
    final VectorSimilarityFunction vectorSimilarityFunction;

    BulkQuantizedCentroidQueryScorer(
        int centroidCount,
        VectorSimilarityFunction vectorSimilarityFunction,
        IndexInput centroids,
        float globalCentroidDp,
        float[] targetQuery,
        byte[] quantizedTargetQuery,
        OptimizedScalarQuantizer.QuantizationResult targetQueryCorrections) {
      this.vectorSimilarityFunction = vectorSimilarityFunction;
      this.centroidCount = centroidCount;
      this.centroids = centroids;
      this.globalCentroidDp = globalCentroidDp;
      this.targetQuery = targetQuery;
      this.quantizedTargetQuery = quantizedTargetQuery;
      this.targetQueryCorrections = targetQueryCorrections;
      this.bulkCentroidLoopBound =
          centroidCount - Math.floorMod(centroidCount, OSQVectorsScorer.BULK_SIZE);
      this.centroidByteSize = IVFUtils.calculateByteLength(targetQuery.length, (byte) 1);
      this.centroid = new float[targetQuery.length];
    }

    @Override
    public float[] centroid(int centroidOrdinal) throws IOException {
      if (centroidOrdinal == currentCentroid) {
        return centroid;
      }
      centroids.seek(
          (long) centroidCount * centroidByteSize
              + (long) Float.BYTES * targetQuery.length * centroidOrdinal);
      centroids.readFloats(centroid, 0, centroid.length);
      currentCentroid = centroidOrdinal;
      return centroid;
    }

    @Override
    public IVFUtils.IntIterator centroidIterator() throws IOException {
      NeighborQueue neighborQueue = new NeighborQueue(centroidCount, true);
      OSQVectorsScorer osqScorer =
          VECTORIZATION_PROVIDER.newOSQVectorsScorer(centroids, targetQuery.length);
      // iterate centroids and score by BULK_SIZE
      int centroidIdx = 0;
      float[] scores = new float[OSQVectorsScorer.BULK_SIZE];
      for (; centroidIdx < bulkCentroidLoopBound; centroidIdx += OSQVectorsScorer.BULK_SIZE) {
        osqScorer.scoreBulk(
            quantizedTargetQuery,
            targetQueryCorrections,
            vectorSimilarityFunction,
            globalCentroidDp,
            OSQVectorsScorer.BULK_SIZE,
            scores);
        for (int i = 0; i < OSQVectorsScorer.BULK_SIZE; i++) {
          neighborQueue.add(centroidIdx + i, scores[i]);
        }
      }
      int centroidsLeft = centroidCount - centroidIdx;
      if (centroidsLeft > 0) {
        osqScorer.scoreBulk(
            quantizedTargetQuery,
            targetQueryCorrections,
            vectorSimilarityFunction,
            globalCentroidDp,
            centroidsLeft,
            scores);
        for (int i = 0; i < centroidsLeft; i++) {
          neighborQueue.add(centroidIdx + i, scores[i]);
        }
      }
      IndexInput unquantizedCentroids =
          centroids.slice(
              "unquantizedCentroids",
              (long) centroidCount * IVFUtils.calculateByteLength(targetQuery.length, (byte) 1),
              (long) centroidCount * targetQuery.length * Float.BYTES);
      return new BulkEstimateIterator(
          vectorSimilarityFunction, targetQuery, neighborQueue, unquantizedCentroids);
    }
  }

  static class BulkEstimateIterator implements IVFUtils.IntIterator {
    final NeighborQueue centroidScoreEstimates;
    final IndexInput centroids;
    final NeighborQueue trueCentroidScores;
    final float[] targetQuery;
    final float[] unquantizedCentroid;
    final VectorSimilarityFunction vectorSimilarityFunction;

    BulkEstimateIterator(
        VectorSimilarityFunction vectorSimilarityFunction,
        float[] targetQuery,
        NeighborQueue centroidScoreEstimates,
        IndexInput centroids) {
      this.centroidScoreEstimates = centroidScoreEstimates;
      this.centroids = centroids;
      // rescore the next 5 at a time
      this.trueCentroidScores = new NeighborQueue(5, true);
      this.vectorSimilarityFunction = vectorSimilarityFunction;
      this.targetQuery = targetQuery;
      this.unquantizedCentroid = new float[targetQuery.length];
    }

    @Override
    public int pop() throws IOException {
      if (trueCentroidScores.size() == 0) {
        rescoreNextBatch();
      }
      return trueCentroidScores.pop();
    }

    @Override
    public boolean hasNext() throws IOException {
      return trueCentroidScores.size() > 0 || centroidScoreEstimates.size() > 0;
    }

    private float scoreCentroid(long ord) throws IOException {
      centroids.seek(ord * targetQuery.length * Float.BYTES);
      centroids.readFloats(unquantizedCentroid, 0, unquantizedCentroid.length);
      return vectorSimilarityFunction.compare(targetQuery, unquantizedCentroid);
    }

    private void rescoreNextBatch() throws IOException {
      assert trueCentroidScores.size() == 0;
      int centroidCount = Math.min(5, centroidScoreEstimates.size());
      int i = 0;
      while (i < centroidCount) {
        int centroidOrdinal = centroidScoreEstimates.pop();
        float score = scoreCentroid(centroidOrdinal);
        trueCentroidScores.add(centroidOrdinal, score);
        i++;
      }
    }
  }

  // TODO can we do this in off-heap blocks?
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
      this.centroidByteSize = IVFUtils.calculateByteLength(dimension, (byte) 1);
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

  private static class MemorySegmentPostingsVisitor extends IVFUtils.PostingVisitor {
    final long quantizedByteLength;
    final IndexInput indexInput;
    final float[] target;
    final FieldEntry entry;
    final FieldInfo fieldInfo;
    final IntPredicate needsScoring;
    private final OSQVectorsScorer osqVectorsScorer;

    int[] docIdsScratch = new int[0];
    int vectors;
    boolean quantized = false;
    float centroidDp;
    float[] centroid;
    long slicePos;
    OptimizedScalarQuantizer.QuantizationResult queryCorrections;

    final float[] scratch;
    final byte[] quantizationScratch;
    final byte[] quantizedQueryScratch;
    final OptimizedScalarQuantizer quantizer;
    final float[] correctiveValues = new float[3];

    MemorySegmentPostingsVisitor(
        float[] target,
        IndexInput indexInput,
        FieldEntry entry,
        FieldInfo fieldInfo,
        IntPredicate needsScoring)
        throws IOException {
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
      osqVectorsScorer =
          VECTORIZATION_PROVIDER.newOSQVectorsScorer(indexInput, fieldInfo.getVectorDimension());
    }

    @Override
    public int resetPostingsScorer(int centroidOrdinal, float[] centroid) throws IOException {
      quantized = false;
      indexInput.seek(entry.postingListOffsets()[centroidOrdinal]);
      vectors = indexInput.readVInt();
      centroidDp = Float.intBitsToFloat(indexInput.readInt());
      this.centroid = centroid;
      // read the doc ids
      docIdsScratch = vectors > docIdsScratch.length ? new int[vectors] : docIdsScratch;
      GroupVIntUtil.readGroupVInts(indexInput, docIdsScratch, vectors);
      prefixSum(docIdsScratch, vectors);
      slicePos = indexInput.getFilePointer();
      return vectors;
    }

    @Override
    public int visit(KnnCollector knnCollector) throws IOException {
      int scoreDocs = 0;
      assert slicePos == indexInput.getFilePointer();
      for (int i = 0; i < vectors; i++) {
        final int docId = docIdsScratch[i];
        if (needsScoring.test(docId) == false) {
          continue;
        }
        quantizeQueryIfNecessary();
        indexInput.seek(slicePos + i * quantizedByteLength);
        final float qcDist = osqVectorsScorer.quantizeScore(quantizedQueryScratch);
        indexInput.readFloats(correctiveValues, 0, correctiveValues.length);
        final int quantizedComponentSum = Short.toUnsignedInt(indexInput.readShort());
        float score =
            osqVectorsScorer.score(
                queryCorrections,
                fieldInfo.getVectorSimilarityFunction(),
                centroidDp,
                correctiveValues[0],
                correctiveValues[1],
                quantizedComponentSum,
                correctiveValues[2],
                qcDist);
        ++scoreDocs;
        knnCollector.incVisitedCount(1);
        knnCollector.collect(docId, score);
        if (knnCollector.earlyTerminated()) {
          return scoreDocs;
        }
      }
      return scoreDocs;
    }

    private void quantizeQueryIfNecessary() {
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
    }
  }
}
