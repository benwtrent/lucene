/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.INDEX_BITS;
import static org.apache.lucene.sandbox.codecs.quantization.IVFVectorsFormat.IVF_VECTOR_COMPONENT;
import static org.apache.lucene.sandbox.codecs.quantization.KMeans.DEFAULT_ITRS;
import static org.apache.lucene.sandbox.codecs.quantization.KMeans.DEFAULT_RESTARTS;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.packAsBinary;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Default implementation of {@link IVFVectorsWriter}. It uses lucene {@link KMeans} algoritm to
 * partition the vector space, and then stores the centroids an posting list in a sequential
 * fashion.
 */
public class DefaultIVFVectorsWriter extends IVFVectorsWriter {

  @SuppressForbidden(reason = "Uses FMA only where fast and carefully contained")
  private static float fma(float a, float b, float c) {
    if (Constants.HAS_FAST_SCALAR_FMA) {
      return Math.fma(a, b, c);
    } else {
      return a * b + c;
    }
  }

  static final float SOAR_LAMBDA = 0f;

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
                Math.min(Math.sqrt(floatVectorValues.size()), maxNumClusters));
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
            DEFAULT_ITRS,
            desiredClusters * 256);
    float[][] centroids = kMeans.centroids();
    // write them
    OptimizedScalarQuantizer osq =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
    byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
    for (int i = 0; i < centroids.length; i++) {
      System.arraycopy(centroids[i], 0, centroidScratch, 0, centroids[i].length);
      OptimizedScalarQuantizer.QuantizationResult quantizedCentroidResults =
          osq.scalarQuantize(centroidScratch, quantizedScratch, (byte) 4, globalCentroid);
      IVFUtils.writeQuantizedValue(centroidOutput, quantizedScratch, quantizedCentroidResults);
    }
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < centroids.length; i++) {
      buffer.asFloatBuffer().put(centroids[i]);
      centroidOutput.writeBytes(buffer.array(), buffer.array().length);
    }
    return new OnHeapCentroidAssignmentScorer(centroids);
  }

  @Override
  protected long[][] buildAndWritePostingsLists(
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
    final long[][] offsetsAndLengths = new long[randomCentroidScorer.size()][];
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
      offsetsAndLengths[i] = new long[2];
      offsetsAndLengths[i][0] = postingsOutput.getFilePointer();
      int size = cluster.size();
      postingsOutput.writeVInt(size);
      postingsOutput.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
      // varint encode the docIds
      int lastDocId = -1;
      docIdBuffer =
          size > docIdBuffer.length ? ArrayUtil.growExact(docIdBuffer, size) : docIdBuffer;
      int[] dumb = new int[size];
      for (int j = 0; j < size; j++) {
        int docId = floatVectorValues.ordToDoc(cluster.get(j));
        dumb[j] = docId;
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
      for (int cidx = 0; cidx < cluster.size(); cidx++) {
        int ord = cluster.get(cidx);
        // write vector
        byte[] binaryValue = binarizedByteVectorValues.vectorValue(ord);
        OptimizedScalarQuantizer.QuantizationResult corrections =
            binarizedByteVectorValues.getCorrectiveTerms(ord);
        IVFUtils.writeQuantizedValue(postingsOutput, binaryValue, corrections);
      }
      offsetsAndLengths[i][1] = postingsOutput.getFilePointer() - offsetsAndLengths[i][0];
    }
    return offsetsAndLengths;
  }

  @Override
  protected IVFUtils.CentroidAssignmentScorer createCentroidScorer(
      IndexInput centroidsInput, int numCentroids, FieldInfo fieldInfo, float[] globalCentroid)
      throws IOException {
    return new OffHeapCentroidAssignmentScorer(centroidsInput, numCentroids, fieldInfo);
  }

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
    int maxNumClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;
    int desiredClusters =
        (int)
            Math.max(
                maxNumClusters / 16.0,
                Math.min(Math.sqrt(floatVectorValues.size()), maxNumClusters));
    // init centroids from merge state
    List<FloatVectorValues> centroidList = new ArrayList<>();
    for (var reader : mergeState.knnVectorsReaders) {
      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
      if (ivfVectorsReader == null) {
        continue;
      }
      centroidList.add(ivfVectorsReader.getCentroids(fieldInfo));
    }
    FloatVectorValues allPreviousCentroids = new FloatVectorValuesConcat(centroidList);
    float[][] initCentroids = null;
    if (allPreviousCentroids.size() < desiredClusters / 2) {
      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
        mergeState.infoStream.message(
            IVF_VECTOR_COMPONENT,
            "Not enough centroids: "
                + allPreviousCentroids.size()
                + " to bootstrap clustering for desired: "
                + desiredClusters);
      }
      // build the lists
    } else if (allPreviousCentroids.size() > desiredClusters) {
      long nanoTime = System.nanoTime();
      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
        mergeState.infoStream.message(
            IVF_VECTOR_COMPONENT,
            "have centroids: " + allPreviousCentroids.size() + "for desired: " + desiredClusters);
      }
      KMeans kMeans =
          new KMeans(
              allPreviousCentroids,
              desiredClusters,
              new Random(42),
              KMeans.KmeansInitializationMethod.PLUS_PLUS,
              null,
              1,
              5);
      initCentroids = kMeans.computeCentroids(false);
      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
        mergeState.infoStream.message(
            IVF_VECTOR_COMPONENT,
            "initCentroids: "
                + (initCentroids == null ? 0 : initCentroids.length)
                + " time ms: "
                + (System.nanoTime() - nanoTime) / 1000000.0);
      }
    }
    // TODO do more optimized assignment
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
            initCentroids == null ? DEFAULT_RESTARTS : 1,
            initCentroids == null ? DEFAULT_ITRS : 5,
            desiredClusters * 64);
    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
          IVF_VECTOR_COMPONENT, "KMeans time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }
    float[][] centroids = kMeans.centroids();
    // write them
    OptimizedScalarQuantizer osq =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
    float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
    for (int i = 0; i < centroids.length; i++) {
      System.arraycopy(centroids[i], 0, centroidScratch, 0, centroids[i].length);
      OptimizedScalarQuantizer.QuantizationResult result =
          osq.scalarQuantize(centroidScratch, quantizedScratch, (byte) 4, globalCentroid);
      IVFUtils.writeQuantizedValue(temporaryCentroidOutput, quantizedScratch, result);
    }
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < centroids.length; i++) {
      buffer.asFloatBuffer().put(centroids[i]);
      temporaryCentroidOutput.writeBytes(buffer.array(), buffer.array().length);
    }
    return centroids.length;
  }

  @Override
  protected long[][] buildAndWritePostingsLists(
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
    final long[][] offsets = new long[centroidAssignmentScorer.size()][];
    OptimizedScalarQuantizer quantizer =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    BinarizedFloatVectorValues binarizedByteVectorValues =
        new BinarizedFloatVectorValues(floatVectorValues, quantizer);
    int[] docIdBuffer = new int[floatVectorValues.size() / centroidAssignmentScorer.size()];
    for (int i = 0; i < centroidAssignmentScorer.size(); i++) {
      float[] centroid = centroidAssignmentScorer.centroid(i);
      binarizedByteVectorValues.centroid = centroid;
      IntArrayList cluster = clusters[i].sort();
      offsets[i] = new long[2];
      // TODO align???
      offsets[i][0] = postingsOutput.getFilePointer();
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
      for (int cidx = 0; cidx < cluster.size(); cidx++) {
        int ord = cluster.get(cidx);
        // write vector
        IVFUtils.writeQuantizedValue(
            postingsOutput,
            binarizedByteVectorValues.vectorValue(ord),
            binarizedByteVectorValues.getCorrectiveTerms(ord));
      }
      offsets[i][1] = postingsOutput.getFilePointer() - offsets[i][0];
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
    // those for assignment
    //  of vector X (and all other vectors within that centroid).
    short numCentroids = (short) scorer.size();
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      scorer.setScoringVector(vector);
      short bestCentroid = 0;
      if (numCentroids > 1) {
        float minSquaredDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          float squareDist = scorer.score(c);
          if (squareDist < minSquaredDist) {
            bestCentroid = c;
            minSquaredDist = squareDist;
          }
        }
      }
      if (clusters[bestCentroid] == null) {
        clusters[bestCentroid] = new IntArrayList(16);
      }
      clusters[bestCentroid].add(docID);
    }
  }

  // TODO Panama Vector
  static float subtractAndDp(float[] v1, float[] v2, float[] dest) {
    float res = 0f;
    for (int i = 0; i < v1.length; i++) {
      dest[i] = (v1[i] - v2[i]);
      res += dest[i] * dest[i];
    }
    return res;
  }

  // TODO, this is garbage slow, needs rewriting for IntArrayList[] clusters
  static void assignCentroidsSOAR(
      short[] primaryDocCentroids,
      short[] secondaryDocCentroids,
      int[] centroidSize,
      boolean normalizeCentroids,
      FloatVectorValues vectors,
      float[][] centroids)
      throws IOException {
    // TODO, can we initialize the vector centroid search space by their own individual centroids?
    //  e.g. find the nearest N centroids for centroid Y containing vector X, and only consider
    // those for assignment
    //  of vector X (and all other vectors within that centroid).
    short numCentroids = (short) centroids.length;
    assert Arrays.stream(centroidSize).allMatch(size -> size == 0);
    float[] centroidResidualScratch = new float[vectors.dimension()];
    float[] centroidDistancesScratch = new float[numCentroids];
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      short bestCentroid = 0;
      short bestSecondaryCentroid = 0;
      if (numCentroids > 1) {
        float minSquaredDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          // TODO: replace with RandomVectorScorer::score possible on quantized vectors
          float squareDist = VectorUtil.squareDistance(centroids[c], vector);
          centroidDistancesScratch[c] = squareDist;
          if (squareDist < minSquaredDist) {
            bestCentroid = c;
            minSquaredDist = squareDist;
          }
        }
        float n1 = subtractAndDp(vector, centroids[bestCentroid], centroidResidualScratch);
        float secondaryMinDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          if (c == bestCentroid) {
            continue;
          }
          float score = centroidDistancesScratch[c];
          if (SOAR_LAMBDA > 0) {
            float sd = 0;
            float proj = 0;
            for (int i = 0; i < vectors.dimension(); i++) {
              float djk = vector[i] - centroids[c][i];
              sd += djk * djk;
              proj += djk * centroidResidualScratch[i];
            }
            score = sd + SOAR_LAMBDA * proj * proj / n1;
          }
          if (score < secondaryMinDist) {
            bestSecondaryCentroid = c;
            secondaryMinDist = score;
          }
        }
      }
      centroidSize[bestCentroid] += 1;
      centroidSize[bestSecondaryCentroid] += 1;
      primaryDocCentroids[docID] = bestCentroid;
      secondaryDocCentroids[docID] = bestSecondaryCentroid;
    }
    if (normalizeCentroids) {
      for (float[] centroid : centroids) {
        VectorUtil.l2normalize(centroid, false);
      }
    }
    assert Arrays.stream(centroidSize).sum() == vectors.size();
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

  // a simple concatenation of a list of FloatVectorValues
  static class FloatVectorValuesConcat extends FloatVectorValues {
    private final List<FloatVectorValues> values;
    private final int size;
    private final int dimension;
    private final int[] offsets;
    int lastOrd = -1;
    int lastIdx = -1;

    public FloatVectorValuesConcat(List<FloatVectorValues> values) {
      this.values = values;
      int size = 0;
      this.offsets = new int[values.size() + 1];
      int dimension = -1;
      for (int i = 0; i < values.size(); i++) {
        FloatVectorValues value = values.get(i);
        if (value == null) {
          continue;
        }
        size += value.size();
        offsets[i + 1] = offsets[i] + value.size();
        if (dimension == -1) {
          dimension = value.dimension();
        } else if (dimension != value.dimension()) {
          throw new IllegalArgumentException("All vectors must have the same dimension");
        }
      }
      this.size = size;
      this.dimension = dimension;
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
      if (ord >= size || ord < 0) {
        throw new IllegalArgumentException("ord: " + ord + " >= size: " + size);
      }
      if (ord == lastOrd) {
        return values.get(lastIdx).vectorValue(ord - offsets[lastIdx]);
      }
      if (lastOrd != -1 && ord >= offsets[lastIdx] && ord < offsets[lastIdx + 1]) {
        return values.get(lastIdx).vectorValue(ord - offsets[lastIdx]);
      }
      int idx = Arrays.binarySearch(offsets, ord);
      if (idx < 0) {
        idx = -idx - 2;
      }
      lastIdx = idx;
      lastOrd = ord;
      return values.get(idx).vectorValue(ord - offsets[idx]);
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public FloatVectorValues copy() throws IOException {
      return this;
    }
  }

  static class OffHeapCentroidAssignmentScorer implements IVFUtils.CentroidAssignmentScorer {
    private final IndexInput centroidsInput;
    private final int numCentroids;
    private final int dimension;
    private final float[] scratch;
    private float[] q;
    private final long centroidByteSize;
    private int currOrd = -1;

    OffHeapCentroidAssignmentScorer(IndexInput centroidsInput, int numCentroids, FieldInfo info) {
      this.centroidsInput = centroidsInput;
      this.numCentroids = numCentroids;
      this.dimension = info.getVectorDimension();
      this.scratch = new float[dimension];
      this.centroidByteSize = IVFUtils.calculateByteLength(dimension, (byte) 4);
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
}
