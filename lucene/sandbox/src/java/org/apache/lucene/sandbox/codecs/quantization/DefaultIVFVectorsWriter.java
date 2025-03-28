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
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Map;
import java.util.TreeMap;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.LinkedList;
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

/**
 * Default implementation of {@link IVFVectorsWriter}. It uses lucene {@link KMeans} algoritm to
 * partition the vector space, and then stores the centroids an posting list in a sequential
 * fashion.
 */
public class DefaultIVFVectorsWriter extends IVFVectorsWriter {

  static final boolean OVERSPILL_ENABLED = false;
  static final float SOAR_LAMBDA = 1.0f;

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
            DEFAULT_ITRS,
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

//
//  @Override
//  protected int calculateAndWriteCentroids(
//    FieldInfo fieldInfo,
//    FloatVectorValues floatVectorValues,
//    IndexOutput temporaryCentroidOutput,
//    MergeState mergeState,
//    float[] globalCentroid)
//    throws IOException {
//    if (floatVectorValues.size() == 0) {
//      return 0;
//    }
//    int maxNumClusters = ((floatVectorValues.size() - 1) / vectorPerCluster) + 1;
//    int desiredClusters =
//      (int)
//        Math.max(
//          maxNumClusters / 16.0,
//          Math.min(Math.sqrt(floatVectorValues.size()), maxNumClusters));
//    // init centroids from merge state
//    List<FloatVectorValues> centroidList = new ArrayList<>();
//    for (var reader : mergeState.knnVectorsReaders) {
//      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
//      if (ivfVectorsReader == null) {
//        continue;
//      }
//      centroidList.add(ivfVectorsReader.getCentroids(fieldInfo));
//    }
//    FloatVectorValues allPreviousCentroids = new FloatVectorValuesConcat(centroidList);
//    float[][] initCentroids = null;
//    if (allPreviousCentroids.size() < desiredClusters / 2) {
//      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//        mergeState.infoStream.message(
//          IVF_VECTOR_COMPONENT,
//          "Not enough centroids: "
//            + allPreviousCentroids.size()
//            + " to bootstrap clustering for desired: "
//            + desiredClusters);
//      }
//      // build the lists
//    } else if (allPreviousCentroids.size() > desiredClusters) {
//      long nanoTime = System.nanoTime();
//      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//        mergeState.infoStream.message(
//          IVF_VECTOR_COMPONENT,
//          "have centroids: " + allPreviousCentroids.size() + "for desired: " + desiredClusters);
//      }
//      KMeans kMeans =
//        new KMeans(
//          allPreviousCentroids,
//          desiredClusters,
//          new Random(42),
//          KMeans.KmeansInitializationMethod.PLUS_PLUS,
//          null,
//          1,
//          5);
//      initCentroids = kMeans.computeCentroids(false);
//      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//        mergeState.infoStream.message(
//          IVF_VECTOR_COMPONENT,
//          "initCentroids: "
//            + (initCentroids == null ? 0 : initCentroids.length)
//            + " time ms: "
//            + (System.nanoTime() - nanoTime) / 1000000.0);
//      }
//    }
//    // TODO do more optimized assignment
//    long nanoTime = System.nanoTime();
//    final KMeans.Results kMeans =
//      KMeans.centroid(
//        floatVectorValues,
//        desiredClusters,
//        false,
//        42L,
//        KMeans.KmeansInitializationMethod.PLUS_PLUS,
//        initCentroids,
//        fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
//        initCentroids == null ? DEFAULT_RESTARTS : 1,
//        initCentroids == null ? DEFAULT_ITRS : 5,
//        desiredClusters * 64);
//    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//      mergeState.infoStream.message(
//        IVF_VECTOR_COMPONENT, "KMeans time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
//    }
//    float[][] centroids = kMeans.centroids();
//    // write them
//    OptimizedScalarQuantizer osq =
//      new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
//    byte[] quantizedScratch = new byte[fieldInfo.getVectorDimension()];
//    float[] centroidScratch = new float[fieldInfo.getVectorDimension()];
//    for (float[] centroid : centroids) {
//      System.arraycopy(centroid, 0, centroidScratch, 0, centroid.length);
//      OptimizedScalarQuantizer.QuantizationResult result =
//        osq.scalarQuantize(centroidScratch, quantizedScratch, (byte) 4, globalCentroid);
//      IVFUtils.writeQuantizedValue(temporaryCentroidOutput, quantizedScratch, result);
//    }
//    final ByteBuffer buffer =
//      ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
//        .order(ByteOrder.LITTLE_ENDIAN);
//    for (float[] centroid : centroids) {
//      buffer.asFloatBuffer().put(centroid);
//      temporaryCentroidOutput.writeBytes(buffer.array(), buffer.array().length);
//    }
//    return centroids.length;
//  }

  // FIXME: write a utility that compares the clusters this generates with an ideal set of clusters after running kmeans brute force

  record FloatCentroidSplits(
    float[][] centroids
  )
  {}

  record SegmentCentroid(
    int segment,
    int centroid
  ){}

  record CentroidPair(
    SegmentCentroid baseSc,
    SegmentCentroid mergedSc,
    int vectorCount
  ){}

//  @Override
  protected int calculateAndWriteCentroidsOld(
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
    return centroids.length;
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

    record CentroidCount(
      int centroidCount,
      int segmentIndex
    ){}

    List<CentroidCount> centroidCountInSegment = new ArrayList<>();
    Map<SegmentCentroid, Integer> centroidsVectorCount = new HashMap<>();
    int segmentCount = 0;  // FIXME: change this name to segemntIndex if not using elsewhere
    for (var reader : mergeState.knnVectorsReaders) {
      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
      if (ivfVectorsReader == null) {
        continue;
      }

      FloatVectorValues centroid = ivfVectorsReader.getCentroids(fieldInfo);
      centroidList.add(centroid);
      System.out.println("== check z:" + segmentCount);
      centroidCountInSegment.add(new CentroidCount(centroid.size(), segmentCount));

      //create a map of the sizes of the vectors attached to this centroid
      for (int i = 0; i < centroid.size(); i++) {
        int count = ivfVectorsReader.centroidSize(fieldInfo.name, i);
        centroidsVectorCount.put(new SegmentCentroid(segmentCount, i), count);
      }

      segmentCount++;
    }

    Map<Integer, List<SegmentCentroid>> priorCentroidLookup = new HashMap<>();
    Map<Integer, Integer> mergedCentroidsTotalVectorCount = new HashMap<>();

    // merge clusters in the size order
    // FIXME: better comparator here
    centroidCountInSegment.sort((i, j) -> i.centroidCount < j.centroidCount ? 1 : -1);
    int[] orderedSegments = centroidCountInSegment.stream().map(i -> i.segmentIndex).mapToInt(Integer::intValue).toArray();

    FloatVectorValues baseSegment = centroidList.get(orderedSegments[0]);

    for(int j = 0; j < baseSegment.size(); j++) {
      priorCentroidLookup.put(j, new ArrayList<>());
      mergedCentroidsTotalVectorCount.put(j, centroidsVectorCount.get(new SegmentCentroid(orderedSegments[0], j)));
    }

    for(int i = 0; i < orderedSegments.length; i++) {
      // between the largeset and next largest segments find pairwise the clusters closest to one another
      FloatVectorValues mergingSegment = centroidList.get(orderedSegments[i]);
      List<Integer> candidateCentroids = IntStream.range(0, mergingSegment.size()).boxed().collect(Collectors.toCollection(LinkedList::new));

      // compute distances between all candidates in the segment and pick the median or min distance and that's the threshold for merging into the base centroid below
      // FIXME: could try mean or median instead
      float minimumDistance = Float.MAX_VALUE;
      for(int j = 0; j < mergingSegment.size(); j++) {
        float[] candidate1Value = Arrays.copyOf(mergingSegment.vectorValue(j), mergingSegment.dimension());
        for(int k = j+1; k < mergingSegment.size(); k++) {
          float[] candidate2Value = Arrays.copyOf(mergingSegment.vectorValue(k), mergingSegment.dimension());

          float d = VectorUtil.squareDistance(candidate1Value, candidate2Value);
          if(d < minimumDistance) {
            minimumDistance = d;
          }
        }
      }

      for(int j = 0; j < baseSegment.size(); j++) {
        float[] baseCentroid = Arrays.copyOf(baseSegment.vectorValue(j), baseSegment.dimension());

        int closest = -1;

        // check for if we have run out of candidates to merge with and should just keep this current cluster by itself
        if(!candidateCentroids.isEmpty()) {
          // FIXME: need to respect similarity function?? .. don't think so?
          float closestDistance = minimumDistance;
          for (int k = 0; k < candidateCentroids.size(); k++) {
            float[] mergingCentroid = Arrays.copyOf(mergingSegment.vectorValue(k), mergingSegment.dimension());

            // we are willing to merge centroids only if within the min distance and otherwise we treat them both as lone centroids (essentially just appending them)
            float d = VectorUtil.squareDistance(baseCentroid, mergingCentroid);
            if(d < closestDistance) {
              closestDistance = d;
              closest = k;
            }
          }

          // only merge if we actually found a candidate within the appropriate threshold
          if(closest != -1) {
            // remove the centroid from consideration at this point
            candidateCentroids.remove(closest);

            // store a mapping of pairs of cluster to their new candidate cluster
            SegmentCentroid merging = new SegmentCentroid(orderedSegments[i], closest);
            priorCentroidLookup.get(j).add(merging);
            mergedCentroidsTotalVectorCount.compute(j, (_, v) -> v + centroidsVectorCount.get(merging));
          }
        }
      }
    }

    // break apart the clusters that are the largest in size first
    // in size order for each cluster find the center of mass of each of cluster (avg of the original centroids or avg of vector values?) and then split that newly combined centroid into two nearby centroids
    //  for the two new clusters cut the associated size in half and then repeat this process until all the desiredClusters exist
    Map<Integer, FloatCentroidSplits> splitCentroids = new HashMap<>();

    // we'll use this later to know if we need to finally merge centroids that were untouched
    Map<Integer, Boolean> isMerged = new HashMap<>();
    for(int j = 0; j < baseSegment.size(); j++) {
      isMerged.put(j, false);
    }

    Map<Integer, float[]> loneCentroids = new HashMap<>();

    record SegmentCentroidCount(
      List<SegmentCentroid> segmentCentroids,
      int baseCentroidIndex,
      int totalVectorCount,
      int splitLevel   // the total number of times we have split this centroid
    ) {}

    // FIXME: better comparator here
    Queue<SegmentCentroidCount> centroidsToMerge = new PriorityQueue<>(priorCentroidLookup.size(), (sc1, sc2) -> sc1.totalVectorCount < sc2.totalVectorCount? 1 : -1);

    // find the largest ones and combine those first
    // better order these by largest and loop over them once rather than looping over and over again
    for(Map.Entry<Integer, List<SegmentCentroid>> entry : priorCentroidLookup.entrySet()) {
      int vectorCount = mergedCentroidsTotalVectorCount.get(entry.getKey());
      centroidsToMerge.add(new SegmentCentroidCount(
        entry.getValue(),
        entry.getKey(),
        vectorCount,
        1
      ));
    }

    // we init with the total number of centroids we plan to merge
    int totalCentroidsCount = priorCentroidLookup.size();
    while(totalCentroidsCount < desiredClusters && !centroidsToMerge.isEmpty()) {
      // find the largest ones and combine those first

      SegmentCentroidCount scc = centroidsToMerge.remove();
      List<SegmentCentroid> largestArrayOfCentroids = scc.segmentCentroids;
      int baseCentroidIndex = scc.baseCentroidIndex;
      int largest = scc.totalVectorCount;
      int splitLevel = scc.splitLevel;

      // combine all the centroids in the group and them split them out into just two
      assert largestArrayOfCentroids != null;
      float[] baseCentroid = Arrays.copyOf(baseSegment.vectorValue(baseCentroidIndex), baseSegment.dimension());
      int baseCentroidVectorCount = centroidsVectorCount.get(new SegmentCentroid(orderedSegments[0], baseCentroidIndex));

      if(largestArrayOfCentroids.isEmpty()) {
        // there were no aligned centroids from the prior step this is a lone centroid from the base segment
        loneCentroids.put(baseCentroidIndex, baseCentroid);
        // FIXME: previously we assumed the clusters are relatively close in size and therefore we never split an already split cluster again (probably a bad assumption long term) ... now not sure?
        // mergedCentroidsTotalVectorCount.put(baseCentroidIndex, 0);
      } else {
        // FIXME: previously we assumed the clusters are relatively close in size and therefore we never split an already split cluster again (probably a bad assumption long term) ... now not sure?
        // ... but this effectively causes splitting only once
        // mergedCentroidsTotalVectorCount.put(baseCentroidIndex, 0);

        // FIXME: need to account for being able to split them again? ... split their counts in half and then maybe the split centroids will need splitting as well? or could proactively split them until they are smaller than the smallest centroid?
        // FIXME: counts are not entirely accurrate here and that's not a problem because these can be approximate and that's fine; we'll round down here and prioritize splitting other centroids first if we would have otherwise have had equal counts when rounding up
        //  mergedCentroidsTotalVectorCount.put(baseCentroidIndex, largest / 2);

        // even though this is split for now we keep all the associated vectors with this since there's no reason to prematurely divide now
        splitLevel++;
        centroidsToMerge.add(new SegmentCentroidCount(
          largestArrayOfCentroids,
          baseCentroidIndex,
          largest / splitLevel,
          splitLevel
        ));

        int[] centroidsToCombineVectorCount = new int[largestArrayOfCentroids.size()];
        float[][] centroidsToCombine = new float[largestArrayOfCentroids.size()][];
        for (int i = 0; i < largestArrayOfCentroids.size(); i++) {
          SegmentCentroid sc = largestArrayOfCentroids.get(i);
          FloatVectorValues segment = centroidList.get(sc.segment);
          centroidsToCombine[i] = Arrays.copyOf(segment.vectorValue(sc.centroid), segment.dimension());
          centroidsToCombineVectorCount[i] = centroidsVectorCount.get(new SegmentCentroid(sc.segment, sc.centroid));
        }

        // FIXME: try not splitting at all and just keeping the large centroid (disable the kmeans pass below as well?)
        // FIXME: pull this out and only do a combine and split once with the last split level (measure how often this occurs)
        float[] combinedCentroid = combineCentroidsWeightedAvg(baseCentroid, baseCentroidVectorCount, centroidsToCombine, centroidsToCombineVectorCount);
        System.out.println("combined");
        System.out.println(Arrays.toString(combinedCentroid));
        FloatCentroidSplits splitCentroid = splitCentroid(combinedCentroid, splitLevel);
        splitCentroids.put(baseCentroidIndex, splitCentroid);
        isMerged.put(baseCentroidIndex, true);

        // because we merged all the centroids and then split it we add to our total count
        totalCentroidsCount += 1;
      }
    }

    for(Map.Entry<Integer, Boolean> entry : isMerged.entrySet()) {
      if(!entry.getValue()) {
        float[] baseCentroid = Arrays.copyOf(baseSegment.vectorValue(entry.getKey()), baseSegment.dimension());
        int baseCentroidVectorCount = centroidsVectorCount.get(new SegmentCentroid(orderedSegments[0], entry.getKey()));
        List<SegmentCentroid> centroids = priorCentroidLookup.get(entry.getKey());
        int[] centroidsToCombineVectorCount = new int[centroids.size()];
        float[][] centroidsToCombine = new float[centroids.size()][];
        for(int i = 0; i < centroids.size(); i++) {
          SegmentCentroid sc = centroids.get(i);
          FloatVectorValues segment = centroidList.get(orderedSegments[sc.segment]);
          centroidsToCombine[i] = Arrays.copyOf(segment.vectorValue(sc.centroid), segment.dimension());
          centroidsToCombineVectorCount[i] = centroidsVectorCount.get(new SegmentCentroid(sc.segment, sc.centroid));
        }

        float[] combinedCentroid = combineCentroidsWeightedAvg(baseCentroid, baseCentroidVectorCount, centroidsToCombine, centroidsToCombineVectorCount);
        loneCentroids.put(entry.getKey(), combinedCentroid);
      }
    }

    // compile both lone centroids and merged centroids to get a list of initCentroids at desired size
    assert totalCentroidsCount == desiredClusters;
    float[][] initCentroids = new float[totalCentroidsCount][];

    int mergedCount = 0;
    int initCentroidsIndex = 0;
    for(int i = 0; i < baseSegment.size(); i++) {
      // the combined and subsequently split centroids that correspond to the original centroid in the base segment at this location
      if(isMerged.get(i)) {
        FloatCentroidSplits centroidPair = splitCentroids.get(i);
        for(int j = 0; j < centroidPair.centroids.length; j++) {
          initCentroids[initCentroidsIndex++] = centroidPair.centroids[j];
          System.out.println("merged");
          System.out.println(Arrays.toString(centroidPair.centroids[j]));
        }
        mergedCount++;
      } else {
        initCentroids[initCentroidsIndex++] = loneCentroids.get(i);
//        System.out.println("lone");
//        System.out.println(Arrays.toString(loneCentroids.get(i)));
      }
    }

    // FIXME: need a way to maintain the original mapping ... update KMeans to allow maintaining that mapping

    // FIXME: go update the assignCentroids code to respect that mapping from prior centroid to next centroid (via the scorer?)

//    float[][] initCentroids = null;

    // FIXME: add this back? ... clean this up
//    FloatVectorValues allPreviousCentroids = new FloatVectorValuesConcat(centroidList);
//    if (allPreviousCentroids.size() < desiredClusters / 2) {
//      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//        mergeState.infoStream.message(
//            IVF_VECTOR_COMPONENT,
//            "Not enough centroids: "
//                + allPreviousCentroids.size()
//                + " to bootstrap clustering for desired: "
//                + desiredClusters);
//      }
//      // build the lists
//    } else if (allPreviousCentroids.size() > desiredClusters) {
//      long nanoTime = System.nanoTime();
//      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//        mergeState.infoStream.message(
//            IVF_VECTOR_COMPONENT,
//            "have centroids: " + allPreviousCentroids.size() + "for desired: " + desiredClusters);
//      }
//      KMeans kMeans =
//          new KMeans(
//              allPreviousCentroids,
//              desiredClusters,
//              new Random(42),
//              KMeans.KmeansInitializationMethod.PLUS_PLUS,
//              null,
//              1,
//              5);
//      initCentroids = kMeans.computeCentroids(false);
//      if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//        mergeState.infoStream.message(
//            IVF_VECTOR_COMPONENT,
//            "initCentroids: "
//                + (initCentroids == null ? 0 : initCentroids.length)
//                + " time ms: "
//                + (System.nanoTime() - nanoTime) / 1000000.0);
//      }
//    }
    // TODO do more optimized assignment

    // FIXME: run a custom version of kmeans that adjusts the centroids that were split related to only the sets of vectors that were previously associated with the prior centroids

    // FIXME: compare this kmeans outcome with a lot of iterations with the outcome of the process detailed above; ideally a large run of kmeans is approximated by the above algorithm
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

//    float[][] centroids = initCentroids;

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
    return centroids.length;
  }

  private float[] combineCentroids(float[] baseCentroid, float[][] centroidsToCombine) {
    // FIXME: make this a weight avg based on the total number of vectors associated with each centroid
    if(centroidsToCombine.length == 0) {
      return baseCentroid;
    }
    float[] combinedCentroid = new float[baseCentroid.length];
    for(int i = 0; i < baseCentroid.length; i++) {
      combinedCentroid[i] = baseCentroid[i];
    }
    for(int i = 0; i < centroidsToCombine.length; i++) {
      for (int j = 0; j < baseCentroid.length; j++) {
        combinedCentroid[j] += centroidsToCombine[i][j];
      }
    }
    for(int i = 0; i < baseCentroid.length; i++) {
      combinedCentroid[i] /= centroidsToCombine.length;
    }
    return combinedCentroid;
  }

  private float[] combineCentroidsWeightedAvg(float[] baseCentroid, int baseCentroidVectorCount, float[][] centroidsToCombine, int[] centroidsToCombineVectorCount) {
    // FIXME: make this a weight avg based on the total number of vectors associated with each centroid
    if(centroidsToCombine.length == 0) {
      return baseCentroid;
    }
    float[] combinedCentroid = new float[baseCentroid.length];
    for(int i = 0; i < baseCentroid.length; i++) {
      combinedCentroid[i] = baseCentroid[i];
    }
    for(int i = 0; i < centroidsToCombine.length; i++) {
      for (int j = 0; j < baseCentroid.length; j++) {
        // add in the contribution based on the associated total number of vectors
        combinedCentroid[j] += centroidsToCombine[i][j] * ((float) baseCentroidVectorCount / centroidsToCombineVectorCount[i]);
      }
    }
    for(int i = 0; i < baseCentroid.length; i++) {
      combinedCentroid[i] /= centroidsToCombine.length;
    }
    return combinedCentroid;
  }

  private static final float epsilon = 1f / 1024f;

  private FloatCentroidSplits splitCentroid(float[] centroid, int splitLevel) {
    float[][] splitCentroids = new float[splitLevel][centroid.length];

    for(int i = 0; i < splitCentroids.length; i++) {
      System.arraycopy(centroid, 0, splitCentroids[i], 0, centroid.length);
    }

    for(int i = 0; i < centroid.length; i++) {
      for(int j = 0; j < splitCentroids.length; j++) {
        if(j != i % splitLevel) {
          splitCentroids[j][i] *= 1f - epsilon;
        } else {
          splitCentroids[j][i] *= 1f + epsilon;
        }
      }
    }

    return new FloatCentroidSplits(splitCentroids);
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
    //  those for assignment
    //  of vector X (and all other vectors within that centroid).
    short numCentroids = (short) scorer.size();
    // If soar > 0, then we actually need to apply the projection, otherwise, its just the second
    // nearest centroid
    // we at most will look at the 5 nearest centroids if possible
    int soarClusterCheckCount = Math.min(numCentroids, 5);
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
