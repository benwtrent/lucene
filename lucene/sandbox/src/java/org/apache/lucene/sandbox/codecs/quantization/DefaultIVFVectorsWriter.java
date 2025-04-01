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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Map;
import java.util.Set;
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

  // FIXME: write a utility that compares the clusters this generates with an ideal set of clusters after running kmeans brute force

  record SegmentCentroid(
    int segment,
    int centroid,
    int vectorCount
  ){}

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
    float minimumDistance = Float.MAX_VALUE;
    for(int j = 0; j < baseSegment.size(); j++) {
      for(int k = j+1; k < baseSegment.size(); k++) {
        float[] vector1 = new float[fieldInfo.getVectorDimension()];
        float[] vector2 = new float[fieldInfo.getVectorDimension()];
        System.arraycopy(baseSegment.vectorValue(j), 0, vector1, 0, fieldInfo.getVectorDimension());
        System.arraycopy(baseSegment.vectorValue(k), 0, vector2, 0, fieldInfo.getVectorDimension());
        float d = VectorUtil.squareDistance(vector1, vector2);
        if(d < minimumDistance) {
          minimumDistance = d;
        }
      }
    }

    minimumDistance = 1;

    int[] labels = new int[segmentCentroids.size()];
    // loop over segments
    int clusterIdx = 0;
    float[] vector1 = new float[fieldInfo.getVectorDimension()];
    float[] vector2 = new float[fieldInfo.getVectorDimension()];

    boolean[][] scoreExists = new boolean[segmentCentroids.size()][segmentCentroids.size()];
    float[][] scores = new float[segmentCentroids.size()][segmentCentroids.size()];
    // keep track of all inter-centroid distances,
    // using less than centroid * centroid space (e.g. not keeping track of duplicates)

    // FIXME: take incount total vectors in a collapsed cluster instead of just distance so weighted distance instead? ... seeing lots of clusters collapse into each other nearby particularly with the expanding ring approach
    int totalUniqueLabels = segmentCentroids.size();
    float minimumDistanceMultiplier = 1;
    while(totalUniqueLabels > desiredClusters) {
      boolean labelChanged = false;
      for (int i = 0; i < segmentCentroids.size(); i++) {
        if (labels[i] == 0) {
          clusterIdx += 1;
          labels[i] = clusterIdx;
        }
        SegmentCentroid segmentCentroid = segmentCentroids.get(i);
        // FIXME: could only get these on distance computation which may be slower but more memory efficient
        System.arraycopy(centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid), 0, vector1, 0, fieldInfo.getVectorDimension());
        for(int j = i + 1; j < segmentCentroids.size(); j++) {
          SegmentCentroid toCompare = segmentCentroids.get(j);
          float d;
          if(!scoreExists[i][j]) {
            System.arraycopy(centroidList.get(toCompare.segment).vectorValue(toCompare.centroid), 0, vector2, 0, fieldInfo.getVectorDimension());
            scores[i][j] = VectorUtil.squareDistance(vector1, vector2);
            scoreExists[i][j] = true;
          }
          d = scores[i][j];

          if (d < (minimumDistance * minimumDistanceMultiplier)) {
            labelChanged = true;
            if (labels[j] == 0) {
              labels[j] = labels[i];
            } else {
              int baseLabel = labels[i];
              int labelIdx = i;
              while(baseLabel != (labelIdx+1)) {
                baseLabel = labels[baseLabel-1];
                labelIdx = baseLabel-1;
              }
              for (int k = 0; k < labels.length; k++) {
                if (labels[k] == labels[j]) {
                  labels[k] = baseLabel;
                }
              }
            }
          }
        }
      }
      System.out.println(Arrays.toString(labels));
      // FIXME: should be a way to track this inline instead of computing it every time ... do we really need to though?
      if(labelChanged) {
        totalUniqueLabels = 0;
        for(int k = 0; k < labels.length; k++) {
          if(labels[k] == (k+1)) {
            totalUniqueLabels++;
          }
         }
      }
      System.out.println(" === t: " + totalUniqueLabels);
      minimumDistanceMultiplier += 0.10f;
    }

    float[][] centroids = new float[desiredClusters][fieldInfo.getVectorDimension()];
//    float[][] initCentroids = new float[desiredClusters][fieldInfo.getVectorDimension()];
    int[] sum = new int[desiredClusters];

//    float[][] centroids = new float[desiredClusters][];
    int centroidIdx = 0;
    for (int i = 0; i < labels.length; i++) {
      if (labels[i] == (i + 1)) {
        SegmentCentroid segmentCentroid = segmentCentroids.get(i);
        FloatVectorValues segment = centroidList.get(segmentCentroid.segment());
        float[] vector = segment.vectorValue(segmentCentroid.centroid);
        for (int j = 0; j < vector.length; j++) {
          centroids[centroidIdx][j] += (vector[j] * segmentCentroid.vectorCount);
        }
        sum[centroidIdx] += segmentCentroid.vectorCount;
        centroidIdx++;
        System.out.println(" ==== cdix: " + labels[i] + ": " + (i+1) + ": " + centroidIdx);
      }
      System.out.println(" ==== i: " + (i+1) + ": " + labels[i]);
    }
    for (int i = 0; i < centroids.length; i++) {
      for (int j = 0; j < centroids[i].length; j++) {
        centroids[i][j] /= sum[i];
      }
    }
//
//    for (int i = 0; i < segmentCentroids.size(); i++) {
//      SegmentCentroid segmentCentroid = segmentCentroids.get(i);
//      int label = labels[i];
//      FloatVectorValues segment = centroidList.get(segmentCentroid.segment());
//      float[] vector = segment.vectorValue(segmentCentroid.centroid);
//      for (int j = 0; j < vector.length; j++) {
//        initCentroids[label - 1][j] += (vector[j] * segmentCentroid.vectorCount);
//      }
//      sum[label - 1] += segmentCentroid.vectorCount;
//    }
//    for (int i = 0; i < initCentroids.length; i++) {
//      for (int j = 0; j < initCentroids[i].length; j++) {
//        initCentroids[i][j] /= sum[i];
//      }
//    }
    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
      mergeState.infoStream.message(
        IVF_VECTOR_COMPONENT,
        "Agglomerative cluster time ms: " + ((System.nanoTime() - startTime) / 1000000.0));
      mergeState.infoStream.message(
        IVF_VECTOR_COMPONENT,
        "Gathered initCentroids:" + centroids.length + " for desired: " + desiredClusters);
    }

//    System.out.println(Arrays.toString(initCentroids[0]));
//    System.out.println(" ==== c: " + initCentroids.length);

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
//    final KMeans.Results kMeans =
//      KMeans.cluster(
//        floatVectorValues,
//        desiredClusters,
//        false,
//        42L,
//        KMeans.KmeansInitializationMethod.PLUS_PLUS,
//        initCentroids,
//        fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE,
//        1,
//        5,
//        desiredClusters * 64);
//    if (mergeState.infoStream.isEnabled(IVF_VECTOR_COMPONENT)) {
//      mergeState.infoStream.message(
//        IVF_VECTOR_COMPONENT, "KMeans time ms: " + ((System.nanoTime() - nanoTime) / 1000000.0));
//    }
//    float[][] centroids = kMeans.centroids();


    System.out.println(desiredClusters);
    System.out.println(Arrays.toString(labels));
    System.out.println(labels.length);
    System.out.println(segmentCentroids.size());
    System.out.println(centroidIdx);
    System.out.println(Arrays.toString(centroids[0]));
    System.out.println(Arrays.toString(centroids[centroids.length - 1]));

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

//  @Override
  protected int calculateAndWriteCentroidsMine(
    FieldInfo fieldInfo,
    FloatVectorValues floatVectorValues,
    IndexOutput temporaryCentroidOutput,
    MergeState mergeState,
    float[] globalCentroid) throws IOException {
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
    List<SegmentCentroid> segmentCentroids = new ArrayList<>();

    int segmentIdx = 0;
    for (var reader : mergeState.knnVectorsReaders) {
      IVFVectorsReader ivfVectorsReader = IVFUtils.getIVFReader(reader, fieldInfo.name);
      if (ivfVectorsReader == null) {
        continue;
      }

      FloatVectorValues centroid = ivfVectorsReader.getCentroids(fieldInfo);
      centroidList.add(centroid);

      //create a map of the sizes of the vectors attached to this centroid
      for (int i = 0; i < centroid.size(); i++) {
        int vectorCount = ivfVectorsReader.centroidSize(fieldInfo.name, i);
        SegmentCentroid segmentCentroid = new SegmentCentroid(segmentIdx, i, vectorCount);
        segmentCentroids.add(segmentCentroid);
      }

      segmentIdx++;
    }

    centroidList.sort(Comparator.comparingInt(FloatVectorValues::size).reversed());
    FloatVectorValues baseSegment = centroidList.get(0);
    float minimumDistance = Float.MAX_VALUE;
    for(int j = 0; j < baseSegment.size(); j++) {
      for(int k = j+1; k < baseSegment.size(); k++) {
        float[] vector1 = new float[fieldInfo.getVectorDimension()];
        float[] vector2 = new float[fieldInfo.getVectorDimension()];
        System.arraycopy(baseSegment.vectorValue(j), 0, vector1, 0, fieldInfo.getVectorDimension());
        System.arraycopy(baseSegment.vectorValue(k), 0, vector2, 0, fieldInfo.getVectorDimension());
        float d = VectorUtil.squareDistance(vector1, vector2);
        if(d < minimumDistance) {
          minimumDistance = d;
        }
      }
    }

    int[] labels = new int[segmentCentroids.size()];
    // loop over segments
    int clusterIdx = 0;
    float[] vector1 = new float[fieldInfo.getVectorDimension()];
    float[] vector2 = new float[fieldInfo.getVectorDimension()];

    float[][] scores = new float[segmentCentroids.size()][segmentCentroids.size()];
    // FIXME: this works by progressively expanding the radius of the minDistance but it's slow (there's likely a better way to do this) and collapsing these further seems to overall hurt recall
    //  ... try using the precomputed distances to compute what the min distance should be given all of the clusters?
    int totalUniqueLabels = segmentCentroids.size();
    float minimumDistanceMultiplier = 1;
    while(totalUniqueLabels >= desiredClusters) {
      for(int i = 0; i < segmentCentroids.size(); i++) {
        boolean labelChanged = false;
        if (labels[i] == 0) {
          clusterIdx += 1;
          labels[i] = clusterIdx;
        }
        SegmentCentroid segmentCentroid = segmentCentroids.get(i);
        System.arraycopy(centroidList.get(segmentCentroid.segment).vectorValue(segmentCentroid.centroid), 0, vector1, 0, fieldInfo.getVectorDimension());
        for(int j = i + 1; j < segmentCentroids.size(); j++) {
          SegmentCentroid toCompare = segmentCentroids.get(j);
          System.arraycopy(centroidList.get(toCompare.segment).vectorValue(toCompare.centroid), 0, vector2, 0, fieldInfo.getVectorDimension());
          scores[i][j] = VectorUtil.squareDistance(vector1, vector2);
          float d = scores[i][j];
          if (d < (minimumDistance * minimumDistanceMultiplier)) {
            //          if (d < minimumDistance) {
            labelChanged = true;
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
        if(labelChanged) {
          totalUniqueLabels = (int) Arrays.stream(labels).distinct().count();
          System.out.println(totalUniqueLabels);
        }
      }

      System.out.println(totalUniqueLabels);
      totalUniqueLabels = (int) Arrays.stream(labels).distinct().count();
      minimumDistanceMultiplier += 0.1f;
      System.out.println(minimumDistanceMultiplier);
      System.out.println(totalUniqueLabels);
      System.out.println(desiredClusters);
      System.out.println(minimumDistance);
    }
    float[][] centroids = new float[clusterIdx][fieldInfo.getVectorDimension()];
    int[] counts = new int[clusterIdx];
    int[] sum = new int[clusterIdx];
    for (int i = 0; i < segmentCentroids.size(); i++) {
      SegmentCentroid segmentCentroid = segmentCentroids.get(i);
      int label = labels[i];
      FloatVectorValues segment = centroidList.get(segmentCentroid.segment());
      float[] vector = segment.vectorValue(segmentCentroid.centroid);
      for (int j = 0; j < vector.length; j++) {
        centroids[label - 1][j] += (vector[j] * segmentCentroid.vectorCount);
      }
      sum[label - 1] += segmentCentroid.vectorCount;
      counts[label - 1] += 1;
    }
    for (int i = 0; i < centroids.length; i++) {
      for (int j = 0; j < centroids[i].length; j++) {
        centroids[i][j] /= sum[i];
      }
    }

    // FIXME: need a way to maintain the original mapping
    // FIXME: go update the assignCentroids code to respect that mapping from prior centroid to next centroid (via the scorer?)

    // FIXME: compare this kmeans outcome with a lot of iterations with the outcome of the process detailed above; ideally a large run of kmeans is approximated by the above algorithm

    int[] totalUniqueLabels22 = Arrays.stream(labels).distinct().toArray();
    System.out.println(Arrays.toString(labels));
    System.out.println(Arrays.toString(totalUniqueLabels22));
    System.out.println(totalUniqueLabels22.length);

    float[][] initCentroids = new float[desiredClusters][];
    int desiredIdx = 0;
    for(int i  = 0; i < centroids.length; i++) {
      if((i+1)==labels[i]) {
        initCentroids[desiredIdx] = centroids[i];
        desiredIdx++;
      }
    }

    System.out.println(" ---- c3: " + desiredIdx);
    System.out.println(" ---- c2: " + initCentroids.length);

    // FIXME: bad bad bad
    while(desiredIdx < desiredClusters) {
      initCentroids[desiredIdx] = centroids[centroids.length-1];
      desiredIdx++;
    }

    centroids = initCentroids;

    System.out.println(" ---- c: " + centroids.length);

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
