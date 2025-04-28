/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.INDEX_BITS;
import static org.apache.lucene.sandbox.codecs.quantization.IVFVectorsFormat.IVF_VECTOR_COMPONENT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.packAsBinary;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.IntStream;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.internal.vectorization.OSQVectorsScorer;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
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

  static final int MAXK = 128;
  static final float SOAR_LAMBDA = 1.0f;
  // What percentage of the centroids do we do a second check on for SOAR assignment
  static final float EXT_SOAR_LIMIT_CHECK_RATIO = 0.10f;

  private final int vectorPerCluster;

  private final OptimizedScalarQuantizer.QuantizationResult[] corrections =
      new OptimizedScalarQuantizer.QuantizationResult[OSQVectorsScorer.BULK_SIZE];

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
    writeCentroids(centroids, fieldInfo, globalCentroid, centroidOutput);
    return new OnHeapCentroidAssignmentScorer(centroids);
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
    DocIdsWriter docIdsWriter = new DocIdsWriter();
    for (int i = 0; i < randomCentroidScorer.size(); i++) {
      float[] centroid = randomCentroidScorer.centroid(i);
      binarizedByteVectorValues.centroid = centroid;
      // TODO sort by distance to the centroid
      IntArrayList cluster = clusters[i];
      // TODO align???
      offsets[i] = postingsOutput.getFilePointer();
      int size = cluster.size();
      postingsOutput.writeVInt(size);
      postingsOutput.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
      // TODO we might want to consider putting the docIds in a separate file
      //  to aid with only having to fetch vectors from slower storage when they are required
      //  keeping them in the same file indicates we pull the entire file into cache
      docIdsWriter.writeDocIds(
          j -> floatVectorValues.ordToDoc(cluster.get(j)), cluster.size(), postingsOutput);
      writePostingList(cluster, postingsOutput, binarizedByteVectorValues);
    }
    return offsets;
  }

  private void writePostingList(
      IntArrayList cluster,
      IndexOutput postingsOutput,
      BinarizedFloatVectorValues binarizedByteVectorValues)
      throws IOException {
    int limit = cluster.size() - OSQVectorsScorer.BULK_SIZE + 1;
    int cidx = 0;
    // Write vectors in bulks of OSQVectorsScorer.BULK_SIZE.
    for (; cidx < limit; cidx += OSQVectorsScorer.BULK_SIZE) {
      for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
        int ord = cluster.get(cidx + j);
        byte[] binaryValue = binarizedByteVectorValues.vectorValue(ord);
        // write vector
        postingsOutput.writeBytes(binaryValue, 0, binaryValue.length);
        corrections[j] = binarizedByteVectorValues.getCorrectiveTerms(ord);
      }
      // write corrections
      for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
        postingsOutput.writeInt(Float.floatToIntBits(corrections[j].lowerInterval()));
      }
      for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
        postingsOutput.writeInt(Float.floatToIntBits(corrections[j].upperInterval()));
      }
      for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
        int targetComponentSum = corrections[j].quantizedComponentSum();
        assert targetComponentSum >= 0 && targetComponentSum <= 0xffff;
        postingsOutput.writeShort((short) targetComponentSum);
      }
      for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
        postingsOutput.writeInt(Float.floatToIntBits(corrections[j].additionalCorrection()));
      }
    }
    // write tail
    for (; cidx < cluster.size(); cidx++) {
      int ord = cluster.get(cidx);
      // write vector
      byte[] binaryValue = binarizedByteVectorValues.vectorValue(ord);
      OptimizedScalarQuantizer.QuantizationResult corrections =
          binarizedByteVectorValues.getCorrectiveTerms(ord);
      IVFUtils.writeQuantizedValue(postingsOutput, binaryValue, corrections);
      binarizedByteVectorValues.getCorrectiveTerms(ord);
      postingsOutput.writeBytes(binaryValue, 0, binaryValue.length);
      postingsOutput.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
      postingsOutput.writeInt(Float.floatToIntBits(corrections.upperInterval()));
      postingsOutput.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
      assert corrections.quantizedComponentSum() >= 0
          && corrections.quantizedComponentSum() <= 0xffff;
      postingsOutput.writeShort((short) corrections.quantizedComponentSum());
    }
  }

  @Override
  protected IVFUtils.CentroidAssignmentScorer createCentroidScorer(
      IndexInput centroidsInput, int numCentroids, FieldInfo fieldInfo, float[] globalCentroid)
      throws IOException {
    return new OffHeapCentroidAssignmentScorer(centroidsInput, numCentroids, fieldInfo);
  }

  static void writeCentroids(
      float[][] centroids, FieldInfo fieldInfo, float[] globalCentroid, IndexOutput centroidOutput)
      throws IOException {
    final OptimizedScalarQuantizer osq =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
    float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
    // TODO do we want to store these distances as well for future use?
    float[] distances = new float[centroids.length];
    for (int i = 0; i < centroids.length; i++) {
      distances[i] = VectorUtil.squareDistance(centroids[i], globalCentroid);
    }
    // sort the centroids by distance to globalCentroid, nearest (smallest distance), to furthest
    // (largest)
    for (int i = 0; i < centroids.length; i++) {
      for (int j = i + 1; j < centroids.length; j++) {
        if (distances[i] > distances[j]) {
          float[] tmp = centroids[i];
          centroids[i] = centroids[j];
          centroids[j] = tmp;
          float tmpDistance = distances[i];
          distances[i] = distances[j];
          distances[j] = tmpDistance;
        }
      }
    }
    for (float[] centroid : centroids) {
      System.arraycopy(centroid, 0, centroidScratch, 0, centroid.length);
      OptimizedScalarQuantizer.QuantizationResult result =
          osq.scalarQuantize(centroidScratch, quantizedScratch, (byte) 4, globalCentroid);
      IVFUtils.writeQuantizedValue(centroidOutput, quantizedScratch, result);
    }
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    for (float[] centroid : centroids) {
      buffer.asFloatBuffer().put(centroid);
      centroidOutput.writeBytes(buffer.array(), buffer.array().length);
    }
  }

  public static class KMeansResult {
    public float[][] centroids;
    public short[] assignments;
    public int[] assignmentOrds;
    public float[] assignmentDistances;
    public short[] soarAssignments;
    public float[] soarAssignmentDistances;
    public int iterationsRun;
    public boolean converged;

    public KMeansResult(float[][] centroids, short[] assignments, int[] assignmentOrds,
                        float[] assignmentDistances, short[] soarAssignments,
                        float[] soarAssignmentDistances, int iterationsRun, boolean converged) {
      this.centroids = centroids;
      this.assignments = assignments;
      this.assignmentOrds = assignmentOrds;
      this.assignmentDistances = assignmentDistances;
      this.soarAssignments = soarAssignments;
      this.soarAssignmentDistances = soarAssignmentDistances;
      this.iterationsRun = iterationsRun;
      this.converged = converged;
    }
    public KMeansResult(float[][] centroids, short[] assignments, float[] assignmentDistances) {
      this(centroids, assignments, IntStream.range(0, assignments.length).toArray(), assignmentDistances, null, null, 0, false);
    }
    public KMeansResult() {
      this(new float[0][0], new short[0], new int[0], new float[0], new short[0], new float[0], 0, false);
    }
  }

  public static KMeansResult kMeansHierarchical(FieldInfo fieldInfo, FloatVectorValues vectors, int desiredClusters) throws IOException {
    int maxIterations = 8;
    int samplesPerCluster = 128;
    short clustersPerNeighborhood = 32;
    int depth = 0;

    int targetSize = (int) (vectors.size() / (float) desiredClusters);

    return kMeansHierarchical(fieldInfo, new FloatVectorValuesSlice(vectors), targetSize, maxIterations, samplesPerCluster, clustersPerNeighborhood, depth);
  }

  static KMeansResult kMeansHierarchical(FieldInfo fieldInfo, FloatVectorValuesSlice vectors, int targetSize, int maxIterations, int samplesPerCluster, short clustersPerNeighborhood, int depth) throws IOException {
    int n = vectors.size();

    if (n <= targetSize) {
      return new KMeansResult();
    }

    int k = Math.clamp((int)((n + targetSize / 2.0f) / (float) targetSize), 2, MAXK);
    int m = Math.min(k * samplesPerCluster, vectors.size());

    short[] assignments = new short[vectors.size()];
    float[] assignmentDistances = new float[vectors.size()];

    float[][] centroids;

    long startTime = System.nanoTime();

    final KMeans.Results kMeans =
      KMeans.cluster(
        vectors,
        k,
        false,
        42L,
        KMeans.KmeansInitializationMethod.FORGY,
        null,
        fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
        1,
        maxIterations,
        m);
    centroids = kMeans.centroids();

    // FIXME: remove me
//    System.out.println(" ==== kmeans ms: " + (System.nanoTime() - startTime) / 1000000.0);

    int[] clusterSizes = new int[centroids.length];

    long startTimeKmeans = System.nanoTime();

    for(int i = 0; i < vectors.size(); i++) {
      float smallest = Float.MAX_VALUE;
      short centroidIdx = -1;
      float[] vector = vectors.vectorValue(i);
      for (short j = 0; j < centroids.length; j++) {
        float[] centroid = centroids[j];
        float d = VectorUtil.squareDistance(vector, centroid);
        if (d < smallest) {
          smallest = d;
          centroidIdx = j;
        }
      }
      assignments[i] = centroidIdx;
      assignmentDistances[i] = smallest;
      clusterSizes[centroidIdx]++;
    }

    // FIXME: remove me
//    System.out.println(" ==== assignment ms: " + (System.nanoTime() - startTimeKmeans) / 1000000.0);

    short effectiveK = 0;
    for(int i = 0; i < clusterSizes.length; i++) {
      if(clusterSizes[i] > 0) {
        effectiveK++;
      }
    }

    KMeansResult kMeansResult = new KMeansResult(centroids, assignments, assignmentDistances);

    if (effectiveK == 1) {
      return kMeansResult;
    }

    for (short c = 0; c < clusterSizes.length; c++) {
      // Recurse for each cluster which is larger than targetSize.
      // Give ourselves 30% margin for the target size.
      if (100 * clusterSizes[c] > 134 * targetSize) {
        FloatVectorValuesSlice sample = createClusterSlice(clusterSizes[c], c, vectors, assignments);

        updateAssignmentsWithRecursiveSplit(
          kMeansResult, c, kMeansHierarchical(
            fieldInfo, sample, targetSize,
            maxIterations, samplesPerCluster,
            clustersPerNeighborhood, depth + 1
          )
        );
      }
    }

    if (depth == 0) {
      if (kMeansResult.centroids.length == 1 || kMeansResult.centroids.length >= vectors.size()) {
        kMeansResult = new DefaultIVFVectorsWriter.KMeansResult(kMeansResult.centroids,
          kMeansResult.assignments, kMeansResult.assignmentOrds, kMeansResult.assignmentDistances,
          kMeansResult.soarAssignments, kMeansResult.soarAssignmentDistances,
          0, true);
      } else {
        long startTimeLocalKmeans = System.nanoTime();

        kMeansResult = KMeansLocal.kMeansLocal(vectors, kMeansResult.centroids,
          kMeansResult.assignments, kMeansResult.assignmentOrds, kMeansResult.assignmentDistances,
          clustersPerNeighborhood, maxIterations);

        // FIXME: remove me
//        System.out.println(" ==== local kmeans ms: " + (System.nanoTime() - startTimeLocalKmeans) / 1000000.0);
      }
    }

    return kMeansResult;
  }

  static FloatVectorValuesSlice createClusterSlice(int clusterSize, int cluster, FloatVectorValuesSlice vectors, short[] assignments) {
    int[] slice = new int[clusterSize];
    int idx = 0;
    for(int i = 0; i < assignments.length; i++) {
      if(assignments[i] == cluster) {
        slice[idx] = i;
        idx++;
      }
    }

    return new FloatVectorValuesSlice(vectors, slice);
  }

  static void updateAssignmentsWithRecursiveSplit(KMeansResult current, short cluster, KMeansResult splitClusters) {

    int orgCentroidsSize = current.centroids.length;

    // update based on the outcomes from the split clusters recursion
    if(splitClusters.centroids.length > 1) {
      float[][] newCenters = new float[current.centroids.length +
        splitClusters.centroids.length - 1][current.centroids[0].length];
      System.arraycopy(current.centroids, 0, newCenters, 0, current.centroids.length);

      // replace the original cluster
      short origCentroidOrd = splitClusters.assignments[0];
      newCenters[cluster] = splitClusters.centroids[0];

      // append the remainder
      System.arraycopy(splitClusters.centroids, 1, newCenters, current.centroids.length, splitClusters.centroids.length-1);

      current.centroids = newCenters;

      for(int i = 0; i < splitClusters.assignments.length; i++) {
        // this is a new centroid that was added, and so we'll need to remap it
        if(splitClusters.assignments[i] > origCentroidOrd) {
          int parentOrd = splitClusters.assignmentOrds[i];
          current.assignments[parentOrd] = (short) (splitClusters.assignments[i] + orgCentroidsSize - 1);
          current.assignmentDistances[parentOrd] = splitClusters.assignmentDistances[i];
        } else if(splitClusters.assignments[i] < origCentroidOrd) {
          int parentOrd = splitClusters.assignmentOrds[i];
          current.assignments[parentOrd] = (short) (splitClusters.assignments[i] + orgCentroidsSize);
          current.assignmentDistances[parentOrd] = splitClusters.assignmentDistances[i];
        }
      }
    }
  }  
  
  @Override
  protected Assignments calculateAndWriteCentroids(
    FieldInfo fieldInfo,
    FloatVectorValues floatVectorValues,
    IndexOutput temporaryCentroidOutput,
    MergeState mergeState,
    float[] globalCentroid)
    throws IOException {

    long nanoTime = System.nanoTime();

    if (floatVectorValues.size() == 0) {
      return new Assignments(0, new short[0], new float[0], new short[0], new float[0]);
    }
    int desiredClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;

    // FIXME: remove me
//    System.out.println(" ==== desired clusters: " + desiredClusters);

    KMeansResult kMeansResult = kMeansHierarchical(fieldInfo, floatVectorValues, (int) (desiredClusters * 0.66f));

    // FIXME: remove me
//    System.out.println(" ==== actual clusters: " + kMeansResult.centroids.length);

    float[][] centroids = kMeansResult.centroids;
    short[] assignments = kMeansResult.assignments;
    float[] assignmentDistances = kMeansResult.assignmentDistances;
    short[] soarAssignments = kMeansResult.soarAssignments;
    float[] soarAssignmentDistances = kMeansResult.soarAssignmentDistances;

    // write them
    OptimizedScalarQuantizer osq =
      new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
    float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
    for (float[] centroid : centroids) {
      System.arraycopy(centroid, 0, centroidScratch, 0, centroid.length);
      OptimizedScalarQuantizer.QuantizationResult result =
        osq.scalarQuantize(centroidScratch, quantizedScratch, (byte) 4, globalCentroid);
      IVFUtils.writeQuantizedValue(temporaryCentroidOutput, quantizedScratch, result);
    }
    final ByteBuffer buffer =
      ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN);
    for (float[] centroid : centroids) {
      buffer.asFloatBuffer().put(centroid);
      temporaryCentroidOutput.writeBytes(buffer.array(), buffer.array().length);
    }

    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
        IVF_VECTOR_COMPONENT, "centroid merge and assignment time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
      mergeState.infoStream.message(
        IVF_VECTOR_COMPONENT, "final centroid count: " + centroids.length);
    }

    // FIXME: remove me
//    vectorDistribution(floatVectorValues, centroids);

    return new Assignments(centroids.length, assignments, assignmentDistances, soarAssignments, soarAssignmentDistances);
  }

  private static void vectorDistribution(FloatVectorValues vectors, float[][] centroids) throws IOException {
    // verify that we don't have bad distributions
    Map<Integer, Integer> centroidVectorCounts = new HashMap<>();
    for(int i = 0; i < vectors.size(); i++) {
      float smallest = Float.MAX_VALUE;
      int centroidIdx = -1;
      float[] vector = new float[vectors.dimension()];
      System.arraycopy(vectors.vectorValue(i), 0, vector, 0, vectors.dimension());
      for (int j = 0; j < centroids.length; j++) {
        float[] centroid = centroids[j];
        float d = VectorUtil.squareDistance(vector, centroid);
        if(d < smallest) {
          smallest = d;
          centroidIdx = j;
        }
      }
      centroidVectorCounts.compute(centroidIdx, (_,v) -> v == null ? 1 : v+1);
    }
    System.out.println(" === counts: " + centroidVectorCounts);
  }

  @Override
  protected long[] buildAndWritePostingsLists(
      FieldInfo fieldInfo,
      IVFUtils.CentroidAssignmentScorer centroidAssignmentScorer,
      FloatVectorValues floatVectorValues,
      IndexOutput postingsOutput,
      MergeState mergeState,
      Set<SortedAssignment> sortedAssignments)
      throws IOException {
    IntArrayList[] clusters = new IntArrayList[centroidAssignmentScorer.size()];
    for (int i = 0; i < centroidAssignmentScorer.size(); i++) {
      clusters[i] =
          new IntArrayList(floatVectorValues.size() / centroidAssignmentScorer.size() / 4);
    }
    long nanoTime = System.nanoTime();

    // FIXME: use scorer instead??
    // FIXME: do we guarantee all these get written? see comment two v
    // FIXME: limit total soar assignments as in assignCentroidsMerge
    // FIXME: don't include deleted docs, etc (NO_MORE_DOCS)??
    for(SortedAssignment assignment : sortedAssignments) {
      short c = assignment.centroid();
      if (clusters[c] == null) {
        clusters[c] = new IntArrayList(16);
      }
      clusters[c].add(assignment.docId());
    }

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
    DocIdsWriter docIdsWriter = new DocIdsWriter();
    for (int i = 0; i < centroidAssignmentScorer.size(); i++) {
      float[] centroid = centroidAssignmentScorer.centroid(i);
      binarizedByteVectorValues.centroid = centroid;
      // TODO: sort by distance to the centroid
      IntArrayList cluster = clusters[i];
      // TODO align???
      offsets[i] = postingsOutput.getFilePointer();
      int size = cluster.size();
      postingsOutput.writeVInt(size);
      postingsOutput.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
      // TODO we might want to consider putting the docIds in a separate file
      //  to aid with only having to fetch vectors from slower storage when they are required
      //  keeping them in the same file indicates we pull the entire file into cache
      docIdsWriter.writeDocIds(
          j -> floatVectorValues.ordToDoc(cluster.get(j)), size, postingsOutput);
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
    NeighborQueue neighborsToCheck = new NeighborQueue(soarClusterCheckCount + 1, true);
    OrdScoreIterator ordScoreIterator = new OrdScoreIterator(soarClusterCheckCount + 1);
    float[] scratch = new float[vectors.dimension()];
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      scorer.setScoringVector(vector);
      int bestCentroid = 0;
      float bestScore = Float.MAX_VALUE;
      if (numCentroids > 1) {
        for (short c = 0; c < numCentroids; c++) {
          float squareDist = scorer.score(c);
          neighborsToCheck.insertWithOverflow(c, squareDist);
        }
        // pop the best
        int sz = neighborsToCheck.size();
        int best =
            neighborsToCheck.consumeNodesAndScoresMin(
                ordScoreIterator.ords, ordScoreIterator.scores);
        // TODO yikes....
        ordScoreIterator.idx = sz;
        bestScore = ordScoreIterator.getScore(best);
        bestCentroid = ordScoreIterator.getOrd(best);
      }
      if (clusters[bestCentroid] == null) {
        clusters[bestCentroid] = new IntArrayList(16);
      }
      clusters[bestCentroid].add(docID);
      if (soarClusterCheckCount > 0) {
        assignCentroidSOAR(
            ordScoreIterator,
            docID,
            bestCentroid,
            scorer.centroid(bestCentroid),
            bestScore,
            scratch,
            scorer,
            vectors,
            clusters);
      }
      neighborsToCheck.clear();
    }
  }

  static int prefilterCentroidAssignment(
      int centroidOrd,
      FloatVectorValues segmentCentroids,
      IVFUtils.CentroidAssignmentScorer scorer,
      NeighborQueue neighborsToCheck,
      int[] prefilteredCentroids)
      throws IOException {
    float[] segmentCentroid = segmentCentroids.vectorValue(centroidOrd);
    scorer.setScoringVector(segmentCentroid);
    neighborsToCheck.clear();
    for (short c = 0; c < scorer.size(); c++) {
      float squareDist = scorer.score(c);
      neighborsToCheck.insertWithOverflow(c, squareDist);
    }
    int size = neighborsToCheck.size();
    neighborsToCheck.consumeNodes(prefilteredCentroids);
    return size;
  }

  static void assignCentroidsMerge(
      IVFUtils.CentroidAssignmentScorer scorer,
      FloatVectorValues vectors,
      MergeState state,
      String fieldName,
      IntArrayList[] clusters)
      throws IOException {
    FixedBitSet assigned = new FixedBitSet(vectors.size() + 1);
    short numCentroids = (short) scorer.size();
    // If soar > 0, then we actually need to apply the projection, otherwise, its just the second
    // nearest centroid
    // we at most will look at the EXT_SOAR_LIMIT_CHECK_RATIO nearest centroids if possible
    int soarToCheck = (int) (numCentroids * EXT_SOAR_LIMIT_CHECK_RATIO);
    int soarClusterCheckCount = Math.min(numCentroids - 1, soarToCheck);
    // TODO is this the right to check?
    //   If cluster quality is higher, maybe we can reduce this...
    int prefilteredCentroidCount =
        Math.max(soarClusterCheckCount + 1, numCentroids / state.knnVectorsReaders.length);
    NeighborQueue prefilteredCentroidsToCheck = new NeighborQueue(prefilteredCentroidCount, true);
    NeighborQueue neighborsToCheck = new NeighborQueue(soarClusterCheckCount + 1, true);
    OrdScoreIterator ordScoreIterator = new OrdScoreIterator(soarClusterCheckCount + 1);
    int[] prefilteredCentroids = new int[prefilteredCentroidCount];
    float[] scratch = new float[vectors.dimension()];
    // Can we do a pre-filter by finding the nearest centroids to the original vector centroids?
    for (int idx = 0; idx < state.knnVectorsReaders.length; idx++) {
      KnnVectorsReader reader = state.knnVectorsReaders[idx];
      IVFVectorsReader vectorsReader = getIVFReader(reader, fieldName);
      // No reader, skip
      if (vectorsReader == null) {
        continue;
      }
      MergeState.DocMap docMap = state.docMaps[idx];
      var segmentCentroids = vectorsReader.getCentroids(state.fieldInfos[idx].fieldInfo(fieldName));
      for (int i = 0; i < segmentCentroids.size(); i++) {
        IVFVectorsReader.CentroidInfo info = vectorsReader.centroidVectors(fieldName, i, docMap);
        // Rare, but empty centroid, no point in doing comparisons
        if (info.vectors().size == 0) {
          continue;
        }
        prefilteredCentroidsToCheck.clear();
        int prefiltedCount =
            prefilterCentroidAssignment(
                i, segmentCentroids, scorer, prefilteredCentroidsToCheck, prefilteredCentroids);
        int centroidVectorDocId = -1;
        while ((centroidVectorDocId = info.vectors().nextVectorDocId()) != NO_MORE_DOCS) {
          if (assigned.getAndSet(centroidVectorDocId)) {
            continue;
          }
          neighborsToCheck.clear();
          float[] vector = info.vectors().vectorValue();
          scorer.setScoringVector(vector);
          int bestCentroid;
          float bestScore;
          for (int c = 0; c < prefiltedCount; c++) {
            float squareDist = scorer.score(prefilteredCentroids[c]);
            neighborsToCheck.insertWithOverflow(prefilteredCentroids[c], squareDist);
          }
          int centroidCount = neighborsToCheck.size();
          int best =
              neighborsToCheck.consumeNodesAndScoresMin(
                  ordScoreIterator.ords, ordScoreIterator.scores);
          // yikes
          ordScoreIterator.idx = centroidCount;
          bestScore = ordScoreIterator.getScore(best);
          bestCentroid = ordScoreIterator.getOrd(best);
          if (clusters[bestCentroid] == null) {
            clusters[bestCentroid] = new IntArrayList(16);
          }
          clusters[bestCentroid].add(info.vectors().docId());
          if (soarClusterCheckCount > 0) {
            assignCentroidSOAR(
                ordScoreIterator,
                info.vectors().docId(),
                bestCentroid,
                scorer.centroid(bestCentroid),
                bestScore,
                scratch,
                scorer,
                vectors,
                clusters);
          }
        }
      }
    }

    for (int vecOrd = 0; vecOrd < vectors.size(); vecOrd++) {
      if (assigned.get(vecOrd)) {
        continue;
      }
      float[] vector = vectors.vectorValue(vecOrd);
      scorer.setScoringVector(vector);
      int bestCentroid = 0;
      float bestScore = Float.MAX_VALUE;
      if (numCentroids > 1) {
        for (short c = 0; c < numCentroids; c++) {
          float squareDist = scorer.score(c);
          neighborsToCheck.insertWithOverflow(c, squareDist);
        }
        int centroidCount = neighborsToCheck.size();
        int bestIdx =
            neighborsToCheck.consumeNodesAndScoresMin(
                ordScoreIterator.ords, ordScoreIterator.scores);
        ordScoreIterator.idx = centroidCount;
        bestCentroid = ordScoreIterator.getOrd(bestIdx);
        bestScore = ordScoreIterator.getScore(bestIdx);
      }
      if (clusters[bestCentroid] == null) {
        clusters[bestCentroid] = new IntArrayList(16);
      }
      int docID = vectors.ordToDoc(vecOrd);
      clusters[bestCentroid].add(docID);
      if (soarClusterCheckCount > 0) {
        assignCentroidSOAR(
            ordScoreIterator,
            docID,
            bestCentroid,
            scorer.centroid(bestCentroid),
            bestScore,
            scratch,
            scorer,
            vectors,
            clusters);
      }
      neighborsToCheck.clear();
    }
  }

  static void assignCentroidSOAR(
      OrdScoreIterator centroidsToCheck,
      int docId,
      int bestCentroidId,
      float[] bestCentroid,
      float bestScore,
      float[] scratch,
      IVFUtils.CentroidAssignmentScorer scorer,
      FloatVectorValues vectors,
      IntArrayList[] clusters)
      throws IOException {
    float[] vector = vectors.vectorValue(docId);
    VectorUtil.subtract(vector, bestCentroid, scratch);
    int bestSecondaryCentroid = -1;
    float minDist = Float.MAX_VALUE;
    for (int i = 0; i < centroidsToCheck.size(); i++) {
      float score = centroidsToCheck.getScore(i);
      int centroidOrdinal = centroidsToCheck.getOrd(i);
      if (centroidOrdinal == bestCentroidId) {
        continue;
      }
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

  static class OrdScoreIterator {
    private final int[] ords;
    private final float[] scores;
    private int idx = 0;

    OrdScoreIterator(int size) {
      this.ords = new int[size];
      this.scores = new float[size];
    }

    void add(int ord, float score) {
      ords[idx] = ord;
      scores[idx] = score;
      idx++;
    }

    int getOrd(int idx) {
      return ords[idx];
    }

    float getScore(int idx) {
      return scores[idx];
    }

    void reset() {
      idx = 0;
    }

    int size() {
      return idx;
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
