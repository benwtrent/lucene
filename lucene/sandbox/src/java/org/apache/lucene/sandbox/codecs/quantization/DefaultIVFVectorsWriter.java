/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.INDEX_BITS;
import static org.apache.lucene.sandbox.codecs.quantization.DefaultIVFVectorsReader.VECTORIZATION_PROVIDER;
import static org.apache.lucene.sandbox.codecs.quantization.IVFVectorsFormat.IVF_VECTOR_COMPONENT;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.packAsBinary;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.internal.vectorization.OSQVectorsScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Default implementation of {@link IVFVectorsWriter}. It uses lucene {@link KMeans} algoritm to
 * partition the vector space, and then stores the centroids an posting list in a sequential
 * fashion.
 */
public class DefaultIVFVectorsWriter extends IVFVectorsWriter {

  static final boolean OVERSPILL_ENABLED = true;
  static final float SOAR_LAMBDA = 1.0f;
  // What percentage of the centroids do we do a second check on for SOAR assignment
  static final float EXT_SOAR_LIMIT_CHECK_RATIO = 0.25f;

  private final int vectorPerCluster;

  public DefaultIVFVectorsWriter(
      SegmentWriteState state, FlatVectorsWriter rawVectorDelegate, int vectorPerCluster)
      throws IOException {
    super(state, rawVectorDelegate);
    this.vectorPerCluster = vectorPerCluster;
  }

  @Override
  protected IVFUtils.CentroidAssignmentScorer calculateAndWriteCentroids(
      FieldInfo fieldInfo,
      FloatVectorValues floatVectorValues,
      IndexOutput centroidOutput,
      float[] globalCentroid)
      throws IOException {
    if (floatVectorValues.size() == 0) {
      return new IVFUtils.CentroidAssignmentScorer() {
        @Override
        public int size() {
          return 0;
        }

        @Override
        public float[] centroid(int centroidOrdinal) {
          throw new IllegalStateException("No centroids");
        }

        @Override
        public float score(int centroidOrdinal) {
          throw new IllegalStateException("No centroids");
        }

        @Override
        public void setScoringVector(float[] vector) {
          throw new IllegalStateException("No centroids");
        }
      };
    }
    // calculate the centroids
    int maxNumClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;
    int desiredClusters =
        (int)
            Math.max(
                maxNumClusters / 16.0,
                Math.max(Math.sqrt(floatVectorValues.size()), maxNumClusters));
    if (floatVectorValues.size() / desiredClusters > vectorPerCluster) {
      desiredClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;
    }
    final KMeans.Results kMeans =
        KMeans.cluster(
            floatVectorValues,
            desiredClusters,
            false,
            42L,
            KMeans.KmeansInitializationMethod.PLUS_PLUS,
            null,
            fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
            1,
            15,
            desiredClusters * 256);
    float[][] centroids = kMeans.centroids();
    // write them
    writeCentroids(
        centroidOutput, fieldInfo.getVectorSimilarityFunction(), centroids, globalCentroid);
    return new OnHeapCentroidAssignmentScorer(centroids);
  }

  private void writeCentroids(
      IndexOutput output,
      VectorSimilarityFunction similarityFunction,
      float[][] centroids,
      float[] globalCentroid)
      throws IOException {
    // sort the centroids by distance to globalCentroid
    NeighborQueue queue = new NeighborQueue(centroids.length, false);
    for (int i = 0; i < centroids.length; i++) {
      float[] centroid = centroids[i];
      float d = VectorUtil.squareDistance(globalCentroid, centroid);
      queue.add(i, d);
    }
    float[][] sortedCentroids = new float[centroids.length][];
    for (int i = 0; i < centroids.length; i++) {
      int idx = queue.pop();
      sortedCentroids[i] = centroids[idx];
    }
    centroids = sortedCentroids;
    // write them
    OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(similarityFunction);
    BulkQuantizedVectorsWriter writer =
        new BulkQuantizedVectorsWriter(output, OSQVectorsScorer.BULK_SIZE, globalCentroid, osq);
    for (float[] centroid : centroids) {
      writer.add(centroid);
    }
    writer.finish();
    final ByteBuffer buffer =
        ByteBuffer.allocate(globalCentroid.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float[] centroid : centroids) {
      buffer.asFloatBuffer().put(centroid);
      output.writeBytes(buffer.array(), buffer.array().length);
    }
  }

  @Override
  protected long[] buildAndWritePostingsLists(
      FieldInfo fieldInfo,
      InfoStream infoStream,
      IVFUtils.CentroidAssignmentScorer randomCentroidScorer,
      FloatVectorValues floatVectorValues,
      IndexOutput postingsOutput)
      throws IOException {
    IntArrayList[] clusters = new IntArrayList[randomCentroidScorer.size()];
    for (int i = 0; i < randomCentroidScorer.size(); i++) {
      clusters[i] = new IntArrayList(floatVectorValues.size() / randomCentroidScorer.size() / 4);
    }
    assignCentroids(randomCentroidScorer, floatVectorValues, clusters);
    if (infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      printClusterQualityStatistics(clusters, infoStream);
    }
    // write the posting lists
    final long[] offsets = new long[randomCentroidScorer.size()];
    OptimizedScalarQuantizer quantizer =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    BinarizedFloatVectorValues binarizedByteVectorValues =
        new BinarizedFloatVectorValues(floatVectorValues, quantizer);
    int[] docIdBuffer = new int[floatVectorValues.size() / randomCentroidScorer.size()];
    for (int i = 0; i < randomCentroidScorer.size(); i++) {
      float[] centroid = randomCentroidScorer.centroid(i);
      binarizedByteVectorValues.centroid = centroid;
      IntArrayList cluster = clusters[i].sort();
      // TODO align???
      offsets[i] = postingsOutput.getFilePointer();
      int size = cluster.size();
      postingsOutput.writeVInt(size);
      postingsOutput.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
      // varint encode the docIds
      int lastDocId = -1;
      docIdBuffer =
          size > docIdBuffer.length ? ArrayUtil.growExact(docIdBuffer, size) : docIdBuffer;
      for (int j = 0; j < size; j++) {
        int docId = floatVectorValues.ordToDoc(cluster.get(j));
        assert lastDocId < 0 || docId >= lastDocId;
        if (lastDocId >= 0) {
          docIdBuffer[j] = docId - lastDocId;
        } else {
          docIdBuffer[j] = docId;
        }
        lastDocId = docId;
      }
      // TODO we might want to consider putting the docIds in a separate file
      //  to aid with only having to fetch vectors from slower storage when they are required
      //  keeping them in the same file indicates we pull the entire file into cache
      postingsOutput.writeGroupVInts(docIdBuffer, size);
      writePostingList(cluster, postingsOutput, binarizedByteVectorValues);
    }
    return offsets;
  }

  private void writePostingList(
      IntArrayList cluster,
      IndexOutput postingsOutput,
      BinarizedFloatVectorValues binarizedByteVectorValues)
      throws IOException {
    for (int cidx = 0; cidx < cluster.size(); cidx++) {
      int ord = cluster.get(cidx);
      // write vector
      byte[] binaryValue = binarizedByteVectorValues.vectorValue(ord);
      OptimizedScalarQuantizer.QuantizationResult corrections =
          binarizedByteVectorValues.getCorrectiveTerms(ord);
      IVFUtils.writeQuantizedValue(postingsOutput, binaryValue, corrections);
    }
  }

  @Override
  protected IVFUtils.CentroidAssignmentScorer createCentroidScorer(
      IndexInput centroidsInput, int numCentroids, FieldInfo fieldInfo, float[] globalCentroid)
      throws IOException {
    return new OffHeapCentroidAssignmentScorer(
        centroidsInput, numCentroids, globalCentroid, fieldInfo);
  }

  record SegmentCentroid(int segment, int centroid, int centroidSize) {}

  @Override
  protected int calculateAndWriteCentroids(
      FieldInfo fieldInfo,
      FloatVectorValues floatVectorValues,
      IndexOutput temporaryCentroidOutput,
      MergeState mergeState,
      float[] globalCentroid)
      throws IOException {
    if (floatVectorValues.size() == 0) {
      return 0;
    }
    int desiredClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;
    // init centroids from merge state
    List<FloatVectorValues> centroidList = new ArrayList<>();
    List<SegmentCentroid> segmentCentroids = new ArrayList<>(desiredClusters);

    int segmentIdx = 0;
    long startTime = System.nanoTime();
    for (var reader : mergeState.knnVectorsReaders) {
      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
      if (ivfVectorsReader == null) {
        continue;
      }

      FloatVectorValues centroid = ivfVectorsReader.getCentroids(fieldInfo);
      centroidList.add(centroid);
      for (int i = 0; i < centroid.size(); i++) {
        int size = ivfVectorsReader.centroidSize(fieldInfo.name, i);
        segmentCentroids.add(new SegmentCentroid(segmentIdx, i, size));
      }
      segmentIdx++;
    }

    // merge clusters in the size order
    // sort centroid list by floatvector size
    centroidList.sort(Comparator.comparingInt(FloatVectorValues::size).reversed());
    FloatVectorValues baseSegment = centroidList.get(0);
    float[] scratch = new float[fieldInfo.getVectorDimension()];
    float minimumDistance = Float.MAX_VALUE;
    for (int j = 0; j < baseSegment.size(); j++) {
      System.arraycopy(baseSegment.vectorValue(j), 0, scratch, 0, baseSegment.dimension());
      for (int k = j + 1; k < baseSegment.size(); k++) {
        float d = VectorUtil.squareDistance(scratch, baseSegment.vectorValue(k));
        if (d < minimumDistance) {
          minimumDistance = d;
        }
      }
    }
    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
          IVF_VECTOR_COMPONENT,
          "Agglomerative cluster min distance: "
              + minimumDistance
              + " From biggest segment: "
              + centroidList.get(0).size());
    }
    int[] labels = new int[segmentCentroids.size()];
    // loop over segments
    int clusterIdx = 0;
    // keep track of all inter-centroid distances,
    // using less than centroid * centroid space (e.g. not keeping track of duplicates)
    for (int i = 0; i < segmentCentroids.size(); i++) {
      if (labels[i] == 0) {
        clusterIdx += 1;
        labels[i] = clusterIdx;
      }
      SegmentCentroid segmentCentroid = segmentCentroids.get(i);
      System.arraycopy(
          centroidList.get(segmentCentroid.segment()).vectorValue(segmentCentroid.centroid),
          0,
          scratch,
          0,
          baseSegment.dimension());
      for (int j = i + 1; j < segmentCentroids.size(); j++) {
        float d =
            VectorUtil.squareDistance(
                scratch,
                centroidList
                    .get(segmentCentroids.get(j).segment())
                    .vectorValue(segmentCentroids.get(j).centroid));
        if (d < minimumDistance / 2) {
          if (labels[j] == 0) {
            labels[j] = labels[i];
          } else {
            for (int k = 0; k < labels.length; k++) {
              if (labels[k] == labels[j]) {
                labels[k] = labels[i];
              }
            }
          }
        }
      }
    }
    float[][] initCentroids = new float[clusterIdx][fieldInfo.getVectorDimension()];
    int[] sum = new int[clusterIdx];
    for (int i = 0; i < segmentCentroids.size(); i++) {
      SegmentCentroid segmentCentroid = segmentCentroids.get(i);
      int label = labels[i];
      FloatVectorValues segment = centroidList.get(segmentCentroid.segment());
      float[] vector = segment.vectorValue(segmentCentroid.centroid);
      for (int j = 0; j < vector.length; j++) {
        initCentroids[label - 1][j] += (vector[j] * segmentCentroid.centroidSize);
      }
      sum[label - 1] += segmentCentroid.centroidSize;
    }
    for (int i = 0; i < initCentroids.length; i++) {
      for (int j = 0; j < initCentroids[i].length; j++) {
        initCentroids[i][j] /= sum[i];
      }
    }
    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
          IVF_VECTOR_COMPONENT,
          "Agglomerative cluster time ms: " + ((System.nanoTime() - startTime) / 1000000.0));
      mergeState.infoStream.message(
          IVF_VECTOR_COMPONENT,
          "Gathered initCentroids:" + initCentroids.length + " for desired: " + desiredClusters);
    }

    // FIXME: still split to get to desired cluster count?
    // FIXME: need a way to maintain the original mapping ... update KMeans to allow maintaining
    // that mapping
    // FIXME: go update the assignCentroids code to respect that mapping from prior centroid to next
    // centroid (via the scorer?)
    // FIXME: run a custom version of kmeans that adjusts the centroids that were split related to
    // only the sets of vectors that were previously associated with the prior centroids
    // FIXME: compare this kmeans outcome with a lot of iterations with the outcome of the process
    // detailed above; ideally a large run of kmeans is approximated by the above algorithm
    long nanoTime = System.nanoTime();
    final KMeans.Results kMeans =
        KMeans.cluster(
            floatVectorValues,
            desiredClusters,
            false,
            42L,
            KMeans.KmeansInitializationMethod.PLUS_PLUS,
            initCentroids,
            fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
            1,
            5,
            desiredClusters * 64);
    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
          IVF_VECTOR_COMPONENT, "KMeans time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }
    float[][] centroids = kMeans.centroids();

    // sort the centroids by distance to globalCentroid
    writeCentroids(
        temporaryCentroidOutput,
        fieldInfo.getVectorSimilarityFunction(),
        centroids,
        globalCentroid);
    return centroids.length;
  }

  @Override
  protected long[] buildAndWritePostingsLists(
      FieldInfo fieldInfo,
      IVFUtils.CentroidAssignmentScorer centroidAssignmentScorer,
      FloatVectorValues floatVectorValues,
      IndexOutput postingsOutput,
      MergeState mergeState)
      throws IOException {
    IntArrayList[] clusters = new IntArrayList[centroidAssignmentScorer.size()];
    for (int i = 0; i < centroidAssignmentScorer.size(); i++) {
      clusters[i] =
          new IntArrayList(floatVectorValues.size() / centroidAssignmentScorer.size() / 4);
    }
    long nanoTime = System.nanoTime();
    assignCentroids(centroidAssignmentScorer, floatVectorValues, clusters);
    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
          IVF_VECTOR_COMPONENT,
          "assignCentroids time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }

    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      printClusterQualityStatistics(clusters, mergeState.infoStream);
    }
    // write the posting lists
    final long[] offsets = new long[centroidAssignmentScorer.size()];
    OptimizedScalarQuantizer quantizer =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    BinarizedFloatVectorValues binarizedByteVectorValues =
        new BinarizedFloatVectorValues(floatVectorValues, quantizer);
    int[] docIdBuffer = new int[floatVectorValues.size() / centroidAssignmentScorer.size()];
    for (int i = 0; i < centroidAssignmentScorer.size(); i++) {
      float[] centroid = centroidAssignmentScorer.centroid(i);
      binarizedByteVectorValues.centroid = centroid;
      IntArrayList cluster = clusters[i].sort();
      // TODO align???
      offsets[i] = postingsOutput.getFilePointer();
      int size = cluster.size();
      postingsOutput.writeVInt(size);
      postingsOutput.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
      // varint encode the docIds
      int lastDocId = 0;
      docIdBuffer =
          size > docIdBuffer.length ? ArrayUtil.growExact(docIdBuffer, size) : docIdBuffer;
      for (int j = 0; j < size; j++) {
        int docId = floatVectorValues.ordToDoc(cluster.get(j));
        docIdBuffer[j] = docId - lastDocId;
        lastDocId = docId;
      }
      // TODO we might want to consider putting the docIds in a separate file
      //  to aid with only having to fetch vectors from slower storage when they are required
      //  keeping them in the same file indicates we pull the entire file into cache
      postingsOutput.writeGroupVInts(docIdBuffer, size);
      writePostingList(cluster, postingsOutput, binarizedByteVectorValues);
    }
    return offsets;
  }

  private static void printClusterQualityStatistics(
      IntArrayList[] clusters, InfoStream infoStream) {
    float min = Float.MAX_VALUE;
    float max = Float.MIN_VALUE;
    float mean = 0;
    float m2 = 0;
    // iteratively compute the variance & mean
    int count = 0;
    for (IntArrayList cluster : clusters) {
      count += 1;
      if (cluster == null) {
        continue;
      }
      float delta = cluster.size() - mean;
      mean += delta / count;
      m2 += delta * (cluster.size() - mean);
      min = Math.min(min, cluster.size());
      max = Math.max(max, cluster.size());
    }
    float variance = m2 / (clusters.length - 1);
    infoStream.message(
        IVF_VECTOR_COMPONENT,
        "Centroid count: "
            + clusters.length
            + " min: "
            + min
            + " max: "
            + max
            + " mean: "
            + mean
            + " stdDev: "
            + Math.sqrt(variance)
            + " variance: "
            + variance);
  }

  static void assignCentroids(
      IVFUtils.CentroidAssignmentScorer scorer, FloatVectorValues vectors, IntArrayList[] clusters)
      throws IOException {
    // TODO, can we initialize the vector centroid search space by their own individual centroids?
    //  e.g. find the nearest N centroids for centroid Y containing vector X, and only consider
    //  those for assignment
    //  of vector X (and all other vectors within that centroid).
    short numCentroids = (short) scorer.size();
    // If soar > 0, then we actually need to apply the projection, otherwise, its just the second
    // nearest centroid
    // we at most will look at the EXT_SOAR_LIMIT_CHECK_RATIO nearest centroids if possible
    int soarToCheck = (int) (numCentroids * EXT_SOAR_LIMIT_CHECK_RATIO);
    int soarClusterCheckCount = Math.min(numCentroids - 1, soarToCheck);
    // if lambda is `0`, that just means overspill to the second nearest, so we will only check the
    // second nearest
    if (SOAR_LAMBDA == 0) {
      soarClusterCheckCount = Math.min(1, soarClusterCheckCount);
    }
    if (OVERSPILL_ENABLED == false) {
      soarClusterCheckCount = 0;
    }
    int numEstimates = Math.min(Math.max((soarClusterCheckCount + 1) * 2, 5), numCentroids);
    NeighborQueue neighborsToCheck = new NeighborQueue(soarClusterCheckCount + 1, true);
    float[] scores = new float[soarClusterCheckCount];
    int[] centroids = new int[soarClusterCheckCount];
    float[] scratch = new float[vectors.dimension()];
    IVFUtils.CentroidAssignmentDistanceEstimator estimator = null;
    if (numEstimates < numCentroids) {
      estimator = scorer.getEstimator(vectors);
    }
    scorer.getEstimator(vectors);
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      scorer.setScoringVector(vector);
      int bestCentroid = 0;
      float bestScore = Float.MAX_VALUE;
      if (numCentroids > 1) {
        if (estimator != null) {
          // estimate the nearest centroids
          NeighborQueue estimated = estimator.estimateNearestCentroids(docID, numEstimates);
          // TODO can we choose when to rescore or not?
          while (estimated.size() > 0) {
            int centroid = estimated.pop();
            float squareDist = scorer.score(centroid);
            neighborsToCheck.insertWithOverflow(centroid, squareDist);
          }
        } else {
          for (short c = 0; c < numCentroids; c++) {
            float squareDist = scorer.score(c);
            neighborsToCheck.insertWithOverflow(c, squareDist);
          }
        }
        // pop the best
        for (int i = soarClusterCheckCount - 1; i >= 0; i--) {
          scores[i] = neighborsToCheck.topScore();
          centroids[i] = neighborsToCheck.pop();
        }
        bestScore = neighborsToCheck.topScore();
        bestCentroid = neighborsToCheck.pop();
      }
      if (clusters[bestCentroid] == null) {
        clusters[bestCentroid] = new IntArrayList(16);
      }
      clusters[bestCentroid].add(docID);
      if (soarClusterCheckCount > 0) {
        assignCentroidSOAR(
            docID,
            scorer.centroid(bestCentroid),
            bestScore,
            scratch,
            centroids,
            scores,
            scorer,
            vectors,
            clusters);
      }
      neighborsToCheck.clear();
    }
  }

  static void assignCentroidSOAR(
      int docId,
      float[] bestCentroid,
      float bestScore,
      float[] scratch,
      int[] centroidsToCheck,
      float[] centroidsToCheckScore,
      IVFUtils.CentroidAssignmentScorer scorer,
      FloatVectorValues vectors,
      IntArrayList[] clusters)
      throws IOException {
    float[] vector = vectors.vectorValue(docId);
    VectorUtil.subtract(vector, bestCentroid, scratch);
    int bestSecondaryCentroid = -1;
    float minDist = Float.MAX_VALUE;
    for (int i = 0; i < centroidsToCheck.length; i++) {
      float score = centroidsToCheckScore[i];
      int centroidOrdinal = centroidsToCheck[i];
      if (SOAR_LAMBDA > 0) {
        float proj = VectorUtil.soarResidual(vector, scorer.centroid(centroidOrdinal), scratch);
        score += SOAR_LAMBDA * proj * proj / bestScore;
      }
      if (score < minDist) {
        bestSecondaryCentroid = centroidOrdinal;
        minDist = score;
      }
    }
    if (bestSecondaryCentroid != -1) {
      clusters[bestSecondaryCentroid].add(docId);
    }
  }

  // TODO unify with OSQ format
  static class BinarizedFloatVectorValues {
    private OptimizedScalarQuantizer.QuantizationResult corrections;
    private final byte[] binarized;
    private final byte[] initQuantized;
    private float[] centroid;
    private final FloatVectorValues values;
    private final OptimizedScalarQuantizer quantizer;

    private int lastOrd = -1;

    BinarizedFloatVectorValues(FloatVectorValues delegate, OptimizedScalarQuantizer quantizer) {
      this.values = delegate;
      this.quantizer = quantizer;
      this.binarized = new byte[discretize(delegate.dimension(), 64) / 8];
      this.initQuantized = new byte[delegate.dimension()];
    }

    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int ord) {
      if (ord != lastOrd) {
        throw new IllegalStateException(
            "attempt to retrieve corrective terms for different ord "
                + ord
                + " than the quantization was done for: "
                + lastOrd);
      }
      return corrections;
    }

    public byte[] vectorValue(int ord) throws IOException {
      if (ord != lastOrd) {
        binarize(ord);
        lastOrd = ord;
      }
      return binarized;
    }

    private void binarize(int ord) throws IOException {
      corrections =
          quantizer.scalarQuantize(values.vectorValue(ord), initQuantized, INDEX_BITS, centroid);
      packAsBinary(initQuantized, binarized);
    }
  }

  static class OffHeapCentroidAssignmentScorer implements IVFUtils.CentroidAssignmentScorer {
    private final IndexInput centroidsInput;
    private final int numCentroids;
    private final int dimension;
    private final float[] scratch;
    private final FieldInfo info;
    private float[] q;
    private final long centroidByteSize;
    private int currOrd = -1;
    private float[] globalCentroid;

    OffHeapCentroidAssignmentScorer(
        IndexInput centroidsInput, int numCentroids, float[] globalCentroid, FieldInfo info) {
      this.centroidsInput = centroidsInput;
      this.info = info;
      this.numCentroids = numCentroids;
      this.dimension = info.getVectorDimension();
      this.scratch = new float[dimension];
      this.centroidByteSize = IVFUtils.calculateByteLength(dimension, (byte) 1);
      this.globalCentroid = globalCentroid;
    }

    public BulkQuantizedCentroidDistanceEstimator getEstimator(FloatVectorValues vectorValues)
        throws IOException {
      float[] centroidDps = null;
      if (info.getVectorSimilarityFunction() != VectorSimilarityFunction.EUCLIDEAN) {
        centroidDps = new float[numCentroids];
        for (int i = 0; i < numCentroids; i++) {
          float[] v = centroid(i);
          centroidDps[i] = VectorUtil.dotProduct(v, v);
        }
      }
      return new BulkQuantizedCentroidDistanceEstimator(
          centroidsInput.clone(),
          centroidDps,
          globalCentroid,
          vectorValues.copy(),
          OSQVectorsScorer.BULK_SIZE,
          numCentroids,
          info);
    }

    @Override
    public int size() {
      return numCentroids;
    }

    @Override
    public float[] centroid(int centroidOrdinal) throws IOException {
      if (centroidOrdinal == currOrd) {
        return scratch;
      }
      centroidsInput.seek(
          numCentroids * centroidByteSize + (long) centroidOrdinal * dimension * Float.BYTES);
      centroidsInput.readFloats(scratch, 0, dimension);
      this.currOrd = centroidOrdinal;
      return scratch;
    }

    @Override
    public void setScoringVector(float[] vector) {
      q = vector;
    }

    @Override
    public float score(int centroidOrdinal) throws IOException {
      return VectorUtil.squareDistance(centroid(centroidOrdinal), q);
    }
  }

  // TODO throw away rawCentroids
  static class OnHeapCentroidAssignmentScorer implements IVFUtils.CentroidAssignmentScorer {
    private final float[][] centroids;
    private float[] q;

    OnHeapCentroidAssignmentScorer(float[][] centroids) {
      this.centroids = centroids;
    }

    @Override
    public int size() {
      return centroids.length;
    }

    @Override
    public void setScoringVector(float[] vector) {
      q = vector;
    }

    @Override
    public float[] centroid(int centroidOrdinal) throws IOException {
      return centroids[centroidOrdinal];
    }

    @Override
    public float score(int centroidOrdinal) throws IOException {
      return VectorUtil.squareDistance(centroid(centroidOrdinal), q);
    }
  }

  static class BulkQuantizedCentroidDistanceEstimator
      implements IVFUtils.CentroidAssignmentDistanceEstimator {
    private final int numCentroids;
    private final int dimension;
    private final float[] scratch;
    private final byte[] vectorQuantizationScratch;
    private final byte[] vectorPackedQuantizationScratch;
    private final float[] scores;
    private final FloatVectorValues vectorValues;
    private final VectorSimilarityFunction similarityFunction;
    private final IndexInput centroidsInput;
    private final NeighborQueue results;
    private final float[] globalCentroid;
    private final OptimizedScalarQuantizer quantizer;
    private final float globalCentroidDp;
    private final int bulkCentroidLoopBound;
    private final float[] centroidDps;

    BulkQuantizedCentroidDistanceEstimator(
        IndexInput centroidsInput,
        float[] centroidDps,
        float[] globalCentroid,
        FloatVectorValues vectors,
        int bulkSize,
        int numCentroids,
        FieldInfo info)
        throws IOException {
      this.similarityFunction = info.getVectorSimilarityFunction();
      this.globalCentroid = globalCentroid;
      this.centroidDps = centroidDps;
      this.vectorValues = vectors;
      this.numCentroids = numCentroids;
      this.dimension = info.getVectorDimension();
      this.scores = new float[bulkSize];
      this.scratch = new float[dimension];
      this.centroidsInput = centroidsInput;
      this.vectorQuantizationScratch = new byte[dimension];
      this.vectorPackedQuantizationScratch = new byte[discretize(dimension, 64) / 2];
      this.quantizer = new OptimizedScalarQuantizer(info.getVectorSimilarityFunction());
      this.results = new NeighborQueue(Math.min(bulkSize * 2, numCentroids), true);
      this.globalCentroidDp = VectorUtil.dotProduct(globalCentroid, globalCentroid);
      this.bulkCentroidLoopBound =
          numCentroids - Math.floorMod(numCentroids, OSQVectorsScorer.BULK_SIZE);
    }

    @Override
    public NeighborQueue estimateNearestCentroids(int vectorOrd, int k) throws IOException {
      results.clear();
      float[] vector = vectorValues.vectorValue(vectorOrd);
      final float transformFloatV =
          similarityFunction != VectorSimilarityFunction.EUCLIDEAN
              ? VectorUtil.dotProduct(vector, vector)
              : 0;
      System.arraycopy(vector, 0, scratch, 0, dimension);
      OptimizedScalarQuantizer.QuantizationResult result =
          quantizer.scalarQuantize(scratch, vectorQuantizationScratch, (byte) 4, globalCentroid);
      OptimizedScalarQuantizer.transposeHalfByte(
          vectorQuantizationScratch, vectorPackedQuantizationScratch);
      // iterate centroids and score by BULK_SIZE
      int centroidIdx = 0;
      centroidsInput.seek(0);
      OSQVectorsScorer osqScorer =
          VECTORIZATION_PROVIDER.newOSQVectorsScorer(centroidsInput, dimension);

      for (; centroidIdx < bulkCentroidLoopBound; centroidIdx += OSQVectorsScorer.BULK_SIZE) {
        osqScorer.scoreBulk(
            vectorPackedQuantizationScratch,
            result,
            similarityFunction,
            globalCentroidDp,
            OSQVectorsScorer.BULK_SIZE,
            scores);
        for (int i = 0; i < OSQVectorsScorer.BULK_SIZE; i++) {
          float score = transformScore(scores[i], centroidIdx + i, transformFloatV);
          addToResults(centroidIdx + i, score, k);
        }
      }
      int centroidsLeft = numCentroids - centroidIdx;
      if (centroidsLeft > 0) {
        osqScorer.scoreBulk(
            vectorPackedQuantizationScratch,
            result,
            similarityFunction,
            globalCentroidDp,
            centroidsLeft,
            scores);
        for (int i = 0; i < centroidsLeft; i++) {
          float score = transformScore(scores[i], centroidIdx + i, transformFloatV);
          addToResults(centroidIdx + i, score, k);
        }
      }
      return results;
    }

    private float transformScore(float score, int centroidIdx, float transformFloatV) {
      if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
        score = 1f / score - 1f;
      } else if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
        if (score < 1) {
          score = (score - 1f) / score;
        } else {
          score = score - 1;
        }
      } else {
        score = 2 * score - 1f;
      }
      score += transformFloatV;
      if (centroidDps != null) {
        score += centroidDps[centroidIdx];
      }
      return score;
    }

    private void addToResults(int centroidOrdinal, float score, int desiredSize) {
      if (results.size() < desiredSize) {
        results.add(centroidOrdinal, score);
      } else if (score < results.topScore()) {
        results.pop();
        results.add(centroidOrdinal, score);
      }
    }
  }

  static class BulkQuantizedVectorsWriter {
    private final IndexOutput out;
    private final int dimension;
    private final float[] centroid;
    private final OptimizedScalarQuantizer quantizer;
    // dim size
    private final float[] scratch;
    private final byte[] quantizationScratch;
    // quantized byte size * bulkSize
    private final byte[] bulkQuantizationScratch;
    // bulkSize in length
    private final float[] lowerIntervals;
    private final float[] upperIntervals;
    private final float[] additionalCorrection;
    private final short[] quantizedComponentSum;
    private int currentQuantizedCount = 0;
    private final int bulkSize;
    private final int packedByteLength;
    private boolean finished;

    BulkQuantizedVectorsWriter(
        IndexOutput out, int bulkSize, float[] centroid, OptimizedScalarQuantizer quantizer) {
      this.out = out;
      this.dimension = centroid.length;
      this.centroid = centroid;
      this.quantizer = quantizer;
      this.scratch = new float[dimension];
      this.quantizationScratch = new byte[dimension];
      this.packedByteLength = discretize(dimension, 64) / 8;
      this.bulkQuantizationScratch = new byte[packedByteLength * bulkSize];
      this.lowerIntervals = new float[bulkSize];
      this.upperIntervals = new float[bulkSize];
      this.additionalCorrection = new float[bulkSize];
      this.quantizedComponentSum = new short[bulkSize];
      this.bulkSize = bulkSize;
    }

    public void add(float[] vector) throws IOException {
      if (finished) {
        throw new IllegalStateException("Cannot add more vectors after finish");
      }
      if (currentQuantizedCount == bulkSize) {
        flush();
      }
      System.arraycopy(vector, 0, scratch, 0, dimension);
      OptimizedScalarQuantizer.QuantizationResult result =
          quantizer.scalarQuantize(scratch, quantizationScratch, (byte) 1, centroid);
      OptimizedScalarQuantizer.packAsBinary(
          quantizationScratch, bulkQuantizationScratch, packedByteLength * currentQuantizedCount);
      lowerIntervals[currentQuantizedCount] = result.lowerInterval();
      upperIntervals[currentQuantizedCount] = result.upperInterval();
      additionalCorrection[currentQuantizedCount] = result.additionalCorrection();
      assert result.quantizedComponentSum() >= 0 && result.quantizedComponentSum() <= 0xffff;
      quantizedComponentSum[currentQuantizedCount] = (short) result.quantizedComponentSum();
      currentQuantizedCount++;
    }

    public void finish() throws IOException {
      if (finished) {
        return;
      }
      if (currentQuantizedCount > 0) {
        flush();
      }
      finished = true;
    }

    private void flush() throws IOException {
      if (currentQuantizedCount == 0) {
        return;
      }
      out.writeBytes(bulkQuantizationScratch, packedByteLength * currentQuantizedCount);
      // write float corrections
      final ByteBuffer buffer =
          ByteBuffer.allocate(currentQuantizedCount * Float.BYTES * 3)
              .order(ByteOrder.LITTLE_ENDIAN);
      buffer.asFloatBuffer().put(lowerIntervals, 0, currentQuantizedCount);
      buffer.asFloatBuffer().put(upperIntervals, 0, currentQuantizedCount);
      buffer.asFloatBuffer().put(additionalCorrection, 0, currentQuantizedCount);
      out.writeBytes(buffer.array(), buffer.array().length);
      // write shorts
      // TODO we could pack this better?
      for (int i = 0; i < currentQuantizedCount; i++) {
        out.writeShort(quantizedComponentSum[i]);
      }
      currentQuantizedCount = 0;
    }
  }

  @FunctionalInterface
  interface FloatToFloatFunction {
    float apply(float value);
  }
}
