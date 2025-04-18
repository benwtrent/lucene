/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

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
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import static org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat.INDEX_BITS;
import static org.apache.lucene.sandbox.codecs.quantization.IVFVectorsFormat.IVF_VECTOR_COMPONENT;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.discretize;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.packAsBinary;

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
    OptimizedScalarQuantizer osq =
        new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
    float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
    byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
    for (float[] centroid : centroids) {
      System.arraycopy(centroid, 0, centroidScratch, 0, centroid.length);
      OptimizedScalarQuantizer.QuantizationResult quantizedCentroidResults =
          osq.scalarQuantize(centroidScratch, quantizedScratch, (byte) 4, globalCentroid);
      IVFUtils.writeQuantizedValue(centroidOutput, quantizedScratch, quantizedCentroidResults);
    }
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    for (float[] centroid : centroids) {
      buffer.asFloatBuffer().put(centroid);
      centroidOutput.writeBytes(buffer.array(), buffer.array().length);
    }
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
    return new OffHeapCentroidAssignmentScorer(centroidsInput, numCentroids, fieldInfo);
  }

  record SegmentCentroid(int segment, int centroid, int centroidSize) {}

  public static class KMeansResult {
    public float[][] centroids;
    public short[] assignments;
    public int[] assignmentOrds;
    public int iterationsRun;
    public boolean converged;

    public KMeansResult(float[][] centroids, short[] assignments, int[] assignmentOrds, int iterationsRun, boolean converged) {
      this.centroids = centroids;
      this.assignments = assignments;
      this.assignmentOrds = assignmentOrds;
      this.iterationsRun = iterationsRun;
      this.converged = converged;
    }
    public KMeansResult(float[][] centroids, short[] assignments, int[] assignmentOrds) {
      this(centroids, assignments, assignmentOrds, 0, false);
    }
    public KMeansResult(float[][] centroids, short[] assignments) {
      this(centroids, assignments, IntStream.range(0, assignments.length).toArray(), 0, false);
    }
  }

  public static KMeansResult kMeansHierarchical(FieldInfo fieldInfo, float[][] initialCentroids, FloatVectorValues vectors, int desiredClusters) throws IOException {
    int maxIterations = 8;
    int maxK = desiredClusters;
    int samplesPerCluster = 128;
    int clustersPerNeighborhood = 32;
    int depth = 0;

//    System.out.println(" ==== desired clusters: " + desiredClusters);
//    System.out.println(" ==== vectors size: " + vectors.size());

    int targetSize = (int) (vectors.size() / (float) desiredClusters);

    return kMeansHierarchical(fieldInfo, initialCentroids, new FloatVectorValuesSlice(vectors), targetSize, maxIterations, maxK, samplesPerCluster, clustersPerNeighborhood, depth);
  }

  static KMeansResult kMeansHierarchical(FieldInfo fieldInfo, float[][] initialCentroids, FloatVectorValuesSlice vectors, int targetSize, int maxIterations, int maxK, int samplesPerCluster, int clustersPerNeighborhood, int depth) throws IOException {

    int n = vectors.size();

    if (n <= targetSize) {
      return new KMeansResult(new float[0][0], new short[0], new int[0]);
    }

    long startTime = System.nanoTime();

    int k = Math.clamp((int)((n + targetSize / 2.0f) / (float) targetSize), 2, maxK);
//    System.out.println(" ==== clamp: " + ((n + targetSize - 1) / targetSize) + " :: " + (n + targetSize - 1));
    int m = Math.min(k * samplesPerCluster, vectors.size());

//    System.out.println(" ==== targetSize: " + targetSize);
//    System.out.println(" ==== n: " + n);
//    System.out.println(" ==== k: " + k);
//    System.out.println(" ==== m: " + m);

    // FIXME: utilize agglomerative clusters as initial state if provided instead of doing kmeans?

//    System.out.println(" ==== initial kmeans " + depth);
//    System.out.println(" ==== max iters: " + maxIterations);
//    System.out.println(" ==== vectors size: " + vectors.size());

    float[][] centroids;
    if(initialCentroids == null) {
      final KMeans.Results kMeans =
        KMeans.cluster(
          vectors,
          k,
          true,
          42L,
          KMeans.KmeansInitializationMethod.RESERVOIR_SAMPLING,
          null,
          fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
          1,
          maxIterations,
          m);
      centroids = kMeans.centroids();
    } else {
      centroids = initialCentroids;
    }

//    System.out.println(" ==== timeit kmh 0: " + ((System.nanoTime() - startTime) / 1000000.0f));

    short[] assignments = new short[vectors.size()];

    // FIXME: this is not what the original impl is doing at all; it's moving the centroids as
    //  part of assignment with the vectors not within the sample for one last
    //  iteration, but do we need to do that?
//    int[] clusterSizes = new int[centroids.length];
//    for(int i = 0; i < vectors.size(); i++) {
//      float smallest = Float.MAX_VALUE;
//      short centroidIdx = -1;
//      float[] vector = vectors.vectorValue(i);
//      for (short j = 0; j < centroids.length; j++) {
//        float[] centroid = centroids[j];
//        float d = VectorUtil.squareDistance(vector, centroid);
//        if(d < smallest) {
//          smallest = d;
//          centroidIdx = j;
//        }
//      }
//      assignments[i] = centroidIdx;
//      clusterSizes[centroidIdx]++;
//    }

    // FIXME: this is the slowest step we have ... parallelize it?
    // FIXME: do sampling here instead?
    int[] clusterSizes = new int[centroids.length];
    for(int i = 0; i < vectors.size(); i++) {
      float smallest = Float.MAX_VALUE;
      short centroidIdx = -1;
      float[] vector = vectors.vectorValue(i);
      for (short j = 0; j < centroids.length; j++) {
        float[] centroid = centroids[j];
        float d = VectorUtil.squareDistance(vector, centroid);
        if(d < smallest) {
          smallest = d;
          centroidIdx = j;
        }
      }
      assignments[i] = centroidIdx;
      clusterSizes[centroidIdx]++;
    }


//    System.out.println(" ==== assignments: " + Arrays.toString(assignments));
//    System.out.println(" ==== centroids len: " + centroids.length);

    short effectiveK = 0;
    for(int i = 0; i < clusterSizes.length; i++) {
      if(clusterSizes[i] > 0) {
        effectiveK++;
      }
    }

//    System.out.println(" ==== effk: " + effectiveK);

    if (effectiveK == 1) {
      return new KMeansResult(centroids, assignments);
    }

    KMeansResult kMeansResult = new KMeansResult(centroids, assignments);

//    System.out.println(" ===== clusterSizes: " + Arrays.toString(clusterSizes));

//    System.out.println(" ==== timeit kmh 1: " + ((System.nanoTime() - startTime) / 1000000.0f));

    for (short c = 0; c < clusterSizes.length; c++) {
      // Recurse for each cluster which is larger than targetSize.
      // Give ourselves 30% margin for the target size.
      if (100 * clusterSizes[c] > 134 * targetSize) {
//        System.out.println(" ===== c: " + c);

        FloatVectorValuesSlice sample = createClusterSlice(clusterSizes[c], c, vectors, assignments);

//        System.out.println(" === update assignments " + depth);

        updateAssignmentsWithRecursiveSplit(
          kMeansResult, c, kMeansHierarchical(
            fieldInfo, null, sample, targetSize,
            maxIterations, maxK, samplesPerCluster,
            clustersPerNeighborhood, depth + 1
          )
        );
      }
    }

//    System.out.println(" ==== timeit kmh 2: " + ((System.nanoTime() - startTime) / 1000000.0f));

    if (depth == 0) {
//      System.out.println(" === kmeans local ");
      kMeansResult = KMeansLocal.kMeansLocal(vectors, kMeansResult.centroids,
        kMeansResult.assignments, kMeansResult.assignmentOrds, clustersPerNeighborhood, maxIterations
      );
    }

//    System.out.println(" ==== timeit kmh 3: " + ((System.nanoTime() - startTime) / 1000000.0f));

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

//    System.out.println(" ===== cluster: " + cluster);
//    System.out.println(" ===== orgcensize: " + orgCentroidsSize);
//    System.out.println(" ===== centroids len: " + current.centroids.length);
//    System.out.println(" ===== current assign: " + Arrays.toString(current.assignments));
//    System.out.println(" ===== splitCentroids len: " + splitClusters.centroids.length);
//    System.out.println(" ===== splitCentroids: " + Arrays.toString(splitClusters.assignments));

    // update based on the outcomes from the split clusters recursion
    if(splitClusters.centroids.length > 1) {
      float[][] newCenters = new float[current.centroids.length +
        splitClusters.centroids.length - 1][current.centroids[0].length];
      System.arraycopy(current.centroids, 0, newCenters, 0, current.centroids.length);

      // replace the original cluster
      short origCentroidOrd = splitClusters.assignments[0]; // 1
      newCenters[cluster] = splitClusters.centroids[0]; // 1 @

      // append the remainder
      System.arraycopy(splitClusters.centroids, 1, newCenters, current.centroids.length, splitClusters.centroids.length-1);

      current.centroids = newCenters;

      for(int i = 0; i < splitClusters.assignments.length; i++) {
        // this is a new centroid that was added, and so we'll need to remap it
        if(splitClusters.assignments[i] > origCentroidOrd) {
          int parentOrd = splitClusters.assignmentOrds[i];

//          System.out.println(" ==== sc ao: " + i + " :: " + (splitClusters.assignments[i] + orgCentroidsSize - 2));

          current.assignments[parentOrd] = (short) (splitClusters.assignments[i] + orgCentroidsSize - 1);
        } else if(splitClusters.assignments[i] < origCentroidOrd) {
          int parentOrd = splitClusters.assignmentOrds[i];
          current.assignments[parentOrd] = (short) (splitClusters.assignments[i] + orgCentroidsSize);
        }
      }
//      System.out.println(" ==== rec centers: " + current.centroids.length);
//      System.out.println(" ==== rec assign: " + Arrays.toString(current.assignments));
    }
  }

  // Tom
  @Override
  protected int calculateAndWriteCentroids(
    FieldInfo fieldInfo,
    FloatVectorValues floatVectorValues,
    IndexOutput temporaryCentroidOutput,
    MergeState mergeState,
    float[] globalCentroid)
    throws IOException {

    long nanoTime = System.nanoTime();

    if (floatVectorValues.size() == 0) {
      return 0;
    }

    int desiredClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;
    int targetSize = (int) (floatVectorValues.size() / (float) desiredClusters);
    int k = Math.clamp((int)((floatVectorValues.size() + targetSize / 2.0f) / (float) targetSize), 2, desiredClusters);

//    float[][] centroids = new float[k][];

    // init centroids from merge state
    List<FloatVectorValues> centroidList = new ArrayList<>();
    List<SegmentCentroid> segmentCentroids = new ArrayList<>(k);
    Map<Integer, List<SegmentCentroid>> segmentToCentroid = new HashMap<>();

    int segmentIdx = 0;
    for (var reader : mergeState.knnVectorsReaders) {
      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
      if (ivfVectorsReader == null) {
        continue;
      }

      FloatVectorValues centroid = ivfVectorsReader.getCentroids(fieldInfo);
      centroidList.add(centroid);
      List<SegmentCentroid> segmentOnlyCentroids = new ArrayList<>();
      for (int i = 0; i < centroid.size(); i++) {
        int size = ivfVectorsReader.centroidSize(fieldInfo.name, i);
        SegmentCentroid sc = new SegmentCentroid(segmentIdx, i, size);
        segmentCentroids.add(sc);
        segmentOnlyCentroids.add(sc);
      }
      segmentToCentroid.put(segmentIdx, segmentOnlyCentroids);
      segmentIdx++;
    }

    float minimumDistance = Float.MAX_VALUE;
    float[] vector1 = new float[fieldInfo.getVectorDimension()];
    float[] vector2 = new float[fieldInfo.getVectorDimension()];

    record SegmentCentroidPair (
      SegmentCentroid sc1,
      SegmentCentroid sc2
    ){}

    Map<SegmentCentroidPair, Float> distanceCache = new HashMap<>();
    Map<SegmentCentroid, Integer> segmentCentroidToCentroidIdx = new HashMap<>();

    long startTime = System.nanoTime();

    // keep track of all inter-centroid distances,
    for(int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      List<SegmentCentroid> segmentOnlyCentroids = segmentToCentroid.get(i);
      for(int j = 0; j < segmentOnlyCentroids.size(); j++) {
        SegmentCentroid segmentCentroid = segmentOnlyCentroids.get(j);
        // FIXME: cache these as well?
        System.arraycopy(
          centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid),
          0,
          vector1,
          0,
          fieldInfo.getVectorDimension());
        for(int m = j+1; m < segmentOnlyCentroids.size(); m++) {
          SegmentCentroid toCompare = segmentOnlyCentroids.get(m);
          vector2 = centroidList.get(toCompare.segment).vectorValue(toCompare.centroid);
//          System.arraycopy(
//            centroidList.get(toCompare.segment).vectorValue(toCompare.centroid),
//            0,
//            vector2,
//            0,
//            fieldInfo.getVectorDimension());
          float d = VectorUtil.squareDistance(vector1, vector2);
          distanceCache.put(new SegmentCentroidPair(segmentCentroid, toCompare), d);
          if( d < minimumDistance ) {
            minimumDistance = d;
          }
        }
      }
    }

    segmentCentroids.sort(Comparator.comparingInt(SegmentCentroid::centroidSize));

    Set<SegmentCentroid> discarded = new HashSet<>();

    // loop from smallest to largest and collect the segment centroids to discard
    for(int i = 0; i < segmentCentroids.size(); i++) {
      SegmentCentroid segmentCentroid = segmentCentroids.get(i);
      // merge smallest into largest first
      for(int j = segmentCentroids.size()-1; j > i; j--) {
        SegmentCentroid toCompare = segmentCentroids.get(j);
        Float d = distanceCache.get(new SegmentCentroidPair(segmentCentroid, toCompare));
        if(d == null) {
          System.arraycopy(centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid), 0, vector1, 0, fieldInfo.getVectorDimension());
          vector2 = centroidList.get(toCompare.segment).vectorValue(toCompare.centroid);
          d = VectorUtil.squareDistance(vector1, vector2);
        }
        if( d < minimumDistance ) {
          discarded.add(segmentCentroid);
          break;
        }
      }
    }

    // prune the smalleset until we have just enough to hit desired clusters
    int segmentCentroidIdx = 0;
    while(segmentCentroids.size() - discarded.size() > k) {
      SegmentCentroid segmentCentroid = segmentCentroids.get(segmentCentroidIdx);
      discarded.add(segmentCentroid);
      segmentCentroidIdx++;
    }

    int centroidIdx = 0;
    float[][] centroids = new float[k][];
    for(SegmentCentroid segmentCentroid : segmentCentroids) {
      if(!discarded.contains(segmentCentroid)) {
        float[] v = new float[fieldInfo.getVectorDimension()];
        System.arraycopy(
          centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid),
          0,
          v,
          0,
          fieldInfo.getVectorDimension());
        segmentCentroidToCentroidIdx.put(segmentCentroid, centroidIdx);
        centroids[centroidIdx] = v;
        centroidIdx++;
      }
    }


//    System.out.println(" ==== timeit: " + ((System.nanoTime() - startTime) / 1000000.0f));

    float[][] initialCentroids = centroids;

//    System.out.println(" === starting ");

    // FIXME: consider tom's inclusion of SOAR at this point with spilled points or something
    //  akin to this such that assignments can exist across multiple centroids
    KMeansResult kMeansResult = kMeansHierarchical(fieldInfo, initialCentroids, floatVectorValues, (int) (desiredClusters * 0.66f));
    centroids = kMeansResult.centroids;
//    short[] assignments = result.assignments;

    // write them

//    System.out.println(" ==== timeit 2: " + ((System.nanoTime() - startTime) / 1000000.0f));

//    System.out.println(" === writing ");

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
        IVF_VECTOR_COMPONENT, "centroid merge time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }

//    System.out.println(" === getting distribution on assignments ");

    System.out.println(" ==== final centroid count: " + centroids.length);
    System.out.println(" ==== desired centroid count: " + desiredClusters);
    vectorDistribution(floatVectorValues, centroids);


//    System.out.println(" === done w merge ");

    return centroids.length;
  }

    // Wags
//  @Override
  protected int calculateAndWriteCentroidsWags(
    FieldInfo fieldInfo,
    FloatVectorValues floatVectorValues,
    IndexOutput temporaryCentroidOutput,
    MergeState mergeState,
    float[] globalCentroid)
    throws IOException {

    long nanoTime = System.nanoTime();

    if (floatVectorValues.size() == 0) {
      return 0;
    }

    int desiredClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;

    // init centroids from merge state
    List<FloatVectorValues> centroidList = new ArrayList<>();
    List<SegmentCentroid> segmentCentroids = new ArrayList<>(desiredClusters);
    Map<Integer, List<SegmentCentroid>> segmentToCentroid = new HashMap<>();

    int segmentIdx = 0;
    for (var reader : mergeState.knnVectorsReaders) {
      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
      if (ivfVectorsReader == null) {
        continue;
      }

      FloatVectorValues centroid = ivfVectorsReader.getCentroids(fieldInfo);
      centroidList.add(centroid);
      List<SegmentCentroid> segmentOnlyCentroids = new ArrayList<>();
      for (int i = 0; i < centroid.size(); i++) {
        int size = ivfVectorsReader.centroidSize(fieldInfo.name, i);
        SegmentCentroid sc = new SegmentCentroid(segmentIdx, i, size);
        segmentCentroids.add(sc);
        segmentOnlyCentroids.add(sc);
      }
      segmentToCentroid.put(segmentIdx, segmentOnlyCentroids);
      segmentIdx++;
    }

    float minimumDistance = Float.MAX_VALUE;
    float[] vector1 = new float[fieldInfo.getVectorDimension()];
    float[] vector2 = new float[fieldInfo.getVectorDimension()];

    record SegmentCentroidPair (
      SegmentCentroid sc1,
      SegmentCentroid sc2
    ){}

    Map<SegmentCentroidPair, Float> distanceCache = new HashMap<>();
    Map<SegmentCentroid, Integer> segmentCentroidToCentroidIdx = new HashMap<>();

    // keep track of all inter-centroid distances,
    for(int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      List<SegmentCentroid> segmentOnlyCentroids = segmentToCentroid.get(i);
      for(int j = 0; j < segmentOnlyCentroids.size(); j++) {
        SegmentCentroid segmentCentroid = segmentOnlyCentroids.get(j);
        // FIXME: cache these as well?
        System.arraycopy(
          centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid),
          0,
          vector1,
          0,
          fieldInfo.getVectorDimension());
        for(int k = j+1; k < segmentOnlyCentroids.size(); k++) {
          SegmentCentroid toCompare = segmentOnlyCentroids.get(k);
          System.arraycopy(
            centroidList.get(toCompare.segment).vectorValue(toCompare.centroid),
            0,
            vector2,
            0,
            fieldInfo.getVectorDimension());
          float d = VectorUtil.squareDistance(vector1, vector2);
          distanceCache.put(new SegmentCentroidPair(segmentCentroid, toCompare), d);
          if( d < minimumDistance ) {
            minimumDistance = d;
          }
        }
      }
    }

    segmentCentroids.sort(Comparator.comparingInt(SegmentCentroid::centroidSize));

//    System.out.println(" ==== sc size1: " + segmentCentroids.get(0).centroidSize);
//    System.out.println(" ==== sc size2: " + segmentCentroids.get(1).centroidSize);

    Set<SegmentCentroid> discarded = new HashSet<>();

    // loop from smallest to largest and collect the segment centroids to discard
    for(int i = 0; i < segmentCentroids.size(); i++) {
      SegmentCentroid segmentCentroid = segmentCentroids.get(i);
      // merge smallest into largest first
      for(int j = segmentCentroids.size()-1; j > i; j--) {
        SegmentCentroid toCompare = segmentCentroids.get(j);
        Float d = distanceCache.get(new SegmentCentroidPair(segmentCentroid, toCompare));
        if(d == null) {
          System.arraycopy(
            centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid),
            0,
            vector1,
            0,
            fieldInfo.getVectorDimension());
          System.arraycopy(
            centroidList.get(toCompare.segment).vectorValue(toCompare.centroid),
            0,
            vector2,
            0,
            fieldInfo.getVectorDimension());
          d = VectorUtil.squareDistance(vector1, vector2);
        }
        if( d < minimumDistance ) {
          discarded.add(segmentCentroid);
          break;
        }
      }
    }

    // prune the smalleset until we have just enough to hit desired clusters
    int segmentCentroidIdx = 0;
    while(segmentCentroids.size() - discarded.size() > desiredClusters) {
      SegmentCentroid segmentCentroid = segmentCentroids.get(segmentCentroidIdx);
      discarded.add(segmentCentroid);
      segmentCentroidIdx++;
    }

    int centroidIdx = 0;
    float[][] centroids = new float[desiredClusters][];
    for(SegmentCentroid segmentCentroid : segmentCentroids) {
      if(!discarded.contains(segmentCentroid)) {
        float[] v = new float[fieldInfo.getVectorDimension()];
        System.arraycopy(
          centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid),
          0,
          v,
          0,
          fieldInfo.getVectorDimension());
        segmentCentroidToCentroidIdx.put(segmentCentroid, centroidIdx);
        centroids[centroidIdx] = v;
        centroidIdx++;
      }
    }

    int maxK = Math.round((float) floatVectorValues.size() / (float) desiredClusters);
    float sampleSize = 0.10f;

    int maxIterations = 5;
    Random random = new Random(42L);
    for(int iterations = 0; iterations < maxIterations; iterations++) {

      // select a random sample of vectors to consider each iteration
      int n = floatVectorValues.size();
      int[] randomVectorIdx = new int[n];
      Set<Integer> generated = new HashSet<>();
      int idx = 0;
      // look at 10% of the data
      while(randomVectorIdx.length < n * sampleSize) {
        // Generate a random number between 0 and n
        int r = random.nextInt(n);
        if(generated.add(r)) {
          randomVectorIdx[idx++] = r;
        }
      }

      for(int innerIters = 0; innerIters < 5; innerIters++) {
        Set<Integer> excludedVectors = new HashSet<>();

        for (SegmentCentroid segmentCentroid : segmentCentroids) {
          if(!segmentCentroidToCentroidIdx.containsKey(segmentCentroid)) {
            continue;
          }
          int segment = segmentCentroid.segment;
          int centroid = segmentCentroid.centroid;
          float[] centroidVector = new float[fieldInfo.getVectorDimension()];

          // FIXME: consider using something other than tha closest vector as the starting point
          // find the closest vector to this cluster; use this as a starting point
          float closestDistance = Float.MAX_VALUE;
          float[] closestVector = new float[fieldInfo.getVectorDimension()];
          for (int j = 0; j < randomVectorIdx.length; j++) {
            int vectorIdx = randomVectorIdx[j];
            if (!excludedVectors.contains(vectorIdx)) {
              float[] v = new float[fieldInfo.getVectorDimension()];
              // FIXME: cache distances

              System.arraycopy(floatVectorValues.vectorValue(vectorIdx), 0, v, 0, v.length);
              System.arraycopy(centroidList.get(segment).vectorValue(centroid), 0, centroidVector, 0, centroidVector.length);
              float d = VectorUtil.squareDistance(v, centroidVector);

              if (d < closestDistance) {
                closestDistance = d;
                closestVector = v;
              }
            }
          }

          float[] baseVector = closestVector;

          // from this starting vector find the nearest maxk vectors from the remaining random set
          int maxKVectors = Math.round(sampleSize * maxK);

          record VectorDistance(float distance, int idx, float[] vector) {
          }

          Queue<VectorDistance> closestVectors = new PriorityQueue<>(maxKVectors, Comparator.comparingDouble(VectorDistance::distance).reversed());
          for (int j = 0; j < randomVectorIdx.length; j++) {
            int vectorIdx = randomVectorIdx[j];
            if (!excludedVectors.contains(vectorIdx)) {
              float[] v = new float[fieldInfo.getVectorDimension()];
              // FIXME: cache distances
              System.arraycopy(floatVectorValues.vectorValue(vectorIdx), 0, v, 0, v.length);
              float d = VectorUtil.squareDistance(v, baseVector);

              closestVectors.add(new VectorDistance(d, vectorIdx, v));
              if (closestVectors.size() > maxKVectors) {
                closestVectors.poll();
              }
            }
          }

          // update this centroids position using maxk and then remove all of these vectors from
          //  the set of random vectors in consideration
          int centroidLocalIdx = segmentCentroidToCentroidIdx.get(segmentCentroid);
          for (int j = 0; j < centroids[centroidLocalIdx].length; j++) {
            float newValue = 0f;
            for (VectorDistance vd : closestVectors) {
              newValue += vd.vector[j];
            }
            centroids[centroidLocalIdx][j] = newValue / centroidList.size();
          }

          for (VectorDistance vd : closestVectors) {
            excludedVectors.add(vd.idx);
          }
        }
      }
    }

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
        IVF_VECTOR_COMPONENT, "centroid merge time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }

    vectorDistribution(floatVectorValues, centroids);

    return centroids.length;
  }

//  @Override
  protected int calculateAndWriteCentroidsKmeans(
    FieldInfo fieldInfo,
    FloatVectorValues floatVectorValues,
    IndexOutput temporaryCentroidOutput,
    MergeState mergeState,
    float[] globalCentroid)
    throws IOException {

    long nanoTime = System.nanoTime();

    if (floatVectorValues.size() == 0) {
      return 0;
    }

    int desiredClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;

    float[][] centroids;
    final KMeans.Results kMeans =
      KMeans.cluster(
        floatVectorValues,
        desiredClusters,
        false,
        42L,
        KMeans.KmeansInitializationMethod.PLUS_PLUS,
        null,
        fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
        100,
        10000,
        floatVectorValues.size());
    centroids = kMeans.centroids();

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
        IVF_VECTOR_COMPONENT, "centroid merge time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }

    vectorDistribution(floatVectorValues, centroids);

    return centroids.length;
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

  // Ben's
//  @Override
  protected int calculateAndWriteCentroidsBen(
      FieldInfo fieldInfo,
      FloatVectorValues floatVectorValues,
      IndexOutput temporaryCentroidOutput,
      MergeState mergeState,
      float[] globalCentroid)
      throws IOException {

    long nanoTime = System.nanoTime();

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
//    long nanoTime = System.nanoTime();
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
//    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//      mergeState.infoStream.message(
//          IVF_VECTOR_COMPONENT, "KMeans time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
//    }
    float[][] centroids = kMeans.centroids();

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
        IVF_VECTOR_COMPONENT, "centroid merge time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
    }

    vectorDistribution(floatVectorValues, centroids);

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
    NeighborQueue neighborsToCheck = new NeighborQueue(soarClusterCheckCount + 1, true);
    float[] scores = new float[soarClusterCheckCount];
    int[] centroids = new int[soarClusterCheckCount];
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

  static class NearestCentroidCandidatesProvider {
    IVFVectorsReader currentReader;
    IVFUtils.CentroidAssignmentScorer scorer;
    final int numCandidateCentroids;
    final int[] candidateArray;
    float[] scoreScratch;
    final FieldInfo fieldInfo;
    final NeighborQueue candidateQueue;

    public NearestCentroidCandidatesProvider(
        IVFUtils.CentroidAssignmentScorer scorer, int numCandidateCentroids, FieldInfo fieldInfo) {
      this.scorer = scorer;
      this.numCandidateCentroids = numCandidateCentroids;
      this.candidateArray = new int[numCandidateCentroids];
      this.candidateQueue = new NeighborQueue(numCandidateCentroids, true);
      this.scoreScratch = new float[scorer.size()];
      this.fieldInfo = fieldInfo;
    }

    void setCurrentReader(IVFVectorsReader currentReader) throws IOException {
      this.currentReader = currentReader;
      if (currentReader != null) {
        // gather all the inter centroid scores between currentReader centroids and the scorer
        // centroids
        FloatVectorValues centroids = currentReader.getCentroids(fieldInfo);
        int totalNumberOfScores = centroids.size() * scorer.size();
        if (totalNumberOfScores > scoreScratch.length) {
          scoreScratch = new float[totalNumberOfScores];
        }
        for (int i = 0; i < centroids.size(); i++) {
          float[] vector = centroids.vectorValue(i);
          scorer.setScoringVector(vector);
          for (int j = 0; j < scorer.size(); j++) {
            scoreScratch[j] = scorer.score(j);
          }
        }
      }
    }

    int nearestCentroidCandidates(int[] initialCentroids, int[] results) throws IOException {
      assert results.length == numCandidateCentroids;
      candidateQueue.clear();
      for (int i = 0; i < initialCentroids.length; i++) {
        for (int k = 0; k < scorer.size(); k++) {
          candidateQueue.insertWithOverflow(k, scoreScratch[k * initialCentroids[i]]);
        }
      }
      while (candidateQueue.size() > numCandidateCentroids) {
        candidateQueue.pop();
      }
      int i = 0;
      while (candidateQueue.size() > 0) {
        results[i++] = candidateQueue.pop();
      }
      return i;
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
