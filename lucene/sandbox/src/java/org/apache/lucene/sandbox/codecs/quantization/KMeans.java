/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * with modifications under
 *
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.sandbox.codecs.quantization.SampleReader.createSampleReader;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;

/** KMeans clustering algorithm for vectors */
public class KMeans {
  public static final int MAX_NUM_CENTROIDS = Short.MAX_VALUE; // 32767
  public static final int DEFAULT_RESTARTS = 5;
  public static final int DEFAULT_ITRS = 10;
  public static final int DEFAULT_SAMPLE_SIZE = 100_000;

  private static final float EPS = 1f / 1024f;
  private final FloatVectorValues vectors;
  private final int numVectors;
  private final int numCentroids;
  private final Random random;
  private final KmeansInitializationMethod initializationMethod;
  private final float[][] initCentroids;
  private final int restarts;
  private final int iters;

  /**
   * Cluster vectors into a given number of clusters
   *
   * @param vectors float vectors
   * @param similarityFunction vector similarity function. For COSINE similarity, vectors must be
   *     normalized.
   * @param numClusters number of cluster to cluster vector into
   * @return results of clustering: produced centroids and for each vector its centroid
   * @throws IOException when if there is an error accessing vectors
   */
  public static Results cluster(
      FloatVectorValues vectors, VectorSimilarityFunction similarityFunction, int numClusters)
      throws IOException {
    return cluster(
        vectors,
        numClusters,
        true,
        42L,
        KmeansInitializationMethod.PLUS_PLUS,
        null,
        similarityFunction == VectorSimilarityFunction.COSINE,
        DEFAULT_RESTARTS,
        DEFAULT_ITRS,
        DEFAULT_SAMPLE_SIZE);
  }

  /**
   * Expert: Cluster vectors into a given number of clusters
   *
   * @param vectors float vectors
   * @param numClusters number of cluster to cluster vector into
   * @param assignCentroidsToVectors if {@code true} assign centroids for all vectors. Centroids are
   *     computed on a sample of vectors. If this parameter is {@code true}, in results also return
   *     for all vectors what centroids they belong to.
   * @param seed random seed
   * @param initializationMethod Kmeans initialization method
   * @param initCentroids initial centroids, if not {@code null} utilize as initial centroids for
   *     the given initialization method
   * @param normalizeCenters for cosine distance, set to true, to use spherical k-means where
   *     centers are normalized
   * @param restarts how many times to run Kmeans algorithm
   * @param iters how many iterations to do within a single run
   * @param sampleSize sample size to select from all vectors on which to run Kmeans algorithm
   * @return results of clustering: produced centroids and if {@code assignCentroidsToVectors ==
   *     true} also for each vector its centroid
   * @throws IOException if there is error accessing vectors
   */
  public static Results cluster(
      FloatVectorValues vectors,
      int numClusters,
      boolean assignCentroidsToVectors,
      long seed,
      KmeansInitializationMethod initializationMethod,
      float[][] initCentroids,
      boolean normalizeCenters,
      int restarts,
      int iters,
      int sampleSize)
      throws IOException {
    if (vectors.size() == 0) {
      return null;
    }
    if (numClusters < 1 || numClusters > MAX_NUM_CENTROIDS) {
      throw new IllegalArgumentException(
          "[numClusters] must be between [1] and [" + MAX_NUM_CENTROIDS + "]");
    }
    // adjust sampleSize and numClusters
    sampleSize = Math.max(sampleSize, 100 * numClusters);
    if (sampleSize > vectors.size()) {
      sampleSize = vectors.size();
      // Decrease the number of clusters if needed
      int maxNumClusters = Math.max(1, sampleSize / 100);
      numClusters = Math.min(numClusters, maxNumClusters);
    }

    Random random = new Random(seed);
    float[][] centroids;
    if (numClusters == 1) {
      centroids = new float[1][vectors.dimension()];
      for (int i = 0; i < vectors.size(); i++) {
        float[] vector = vectors.vectorValue(i);
        for (int dim = 0; dim < vector.length; dim++) {
          centroids[0][dim] += vector[dim];
        }
      }
      for (int dim = 0; dim < centroids[0].length; dim++) {
        centroids[0][dim] /= vectors.size();
      }
    } else {
      FloatVectorValues sampleVectors =
          vectors.size() <= sampleSize ? vectors : createSampleReader(vectors, sampleSize, seed);
      KMeans kmeans =
          new KMeans(
              sampleVectors,
              numClusters,
              random,
              initializationMethod,
              initCentroids,
              restarts,
              iters);
      centroids = kmeans.computeCentroids(normalizeCenters);
    }

    short[] vectorCentroids = null;
    int[] centroidSize = null;
    // Assign each vector to the nearest centroid and update the centres
    if (assignCentroidsToVectors) {
      vectorCentroids = new short[vectors.size()];
      centroidSize = new int[centroids.length];
      // Use kahan summation to get more precise results
      assignCentroids(vectorCentroids, centroidSize, vectors, centroids);
    }
    return new Results(centroids, centroidSize, vectorCentroids);
  }

  private static void assignCentroids(
      short[] docCentroids, int[] centroidSize, FloatVectorValues vectors, float[][] centroids)
      throws IOException {
    short numCentroids = (short) centroids.length;
    assert Arrays.stream(centroidSize).allMatch(size -> size == 0);
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      short bestCentroid = 0;
      if (numCentroids > 1) {
        float minSquaredDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          // TODO: replace with RandomVectorScorer::score possible on quantized vectors
          float squareDist = VectorUtil.squareDistance(centroids[c], vector);
          if (squareDist < minSquaredDist) {
            bestCentroid = c;
            minSquaredDist = squareDist;
          }
        }
      }
      centroidSize[bestCentroid] += 1;
      docCentroids[docID] = bestCentroid;
    }

    IntArrayList unassignedCentroids = new IntArrayList();
    for (int c = 0; c < numCentroids; c++) {
      if (centroidSize[c] == 0) {
        unassignedCentroids.add(c);
      }
    }
    if (unassignedCentroids.size() > 0) {
      assignCentroids(vectors, centroids, unassignedCentroids);
    }
    for (int c = 0; c < centroids.length; c++) {
      VectorUtil.l2normalize(centroids[c], false);
    }
    assert Arrays.stream(centroidSize).sum() == vectors.size();
  }

  private final float[] kmeansPlusPlusScratch;

  public KMeans(
      FloatVectorValues vectors,
      int numCentroids,
      Random random,
      KmeansInitializationMethod initializationMethod,
      float[][] initCentroids,
      int restarts,
      int iters) {
    this.vectors = vectors;
    this.numVectors = vectors.size();
    this.numCentroids = numCentroids;
    this.random = random;
    this.initializationMethod = initializationMethod;
    this.restarts = restarts;
    this.iters = iters;
    this.initCentroids = initCentroids;
    this.kmeansPlusPlusScratch =
        initializationMethod == KmeansInitializationMethod.PLUS_PLUS ? new float[numVectors] : null;
  }

  public float[][] computeCentroids(boolean normalizeCenters) throws IOException {
    short[] vectorCentroids = new short[numVectors];
    double minSquaredDist = Double.MAX_VALUE;
    double squaredDist = 0;
    float[][] bestCentroids = null;
    float[][] centroids = new float[numCentroids][vectors.dimension()];
    int restarts = this.restarts;
    int numInitializedCentroids = 0;
    // The user has given us a solid number of centroids to start of with, so skip restarts, fill in
    // where we can, and refine
    if (initCentroids != null && initCentroids.length > numCentroids / 2) {
      int i = 0;
      for (; i < Math.min(numCentroids, initCentroids.length); i++) {
        System.arraycopy(initCentroids[i], 0, centroids[i], 0, initCentroids[i].length);
      }
      numInitializedCentroids = i;
      restarts = 1;
    }

    for (int restart = 0; restart < restarts; restart++) {
      switch (initializationMethod) {
        case FORGY -> initializeForgy(centroids, numInitializedCentroids);
        case RESERVOIR_SAMPLING -> initializeReservoirSampling(centroids, numInitializedCentroids);
        case PLUS_PLUS -> initializePlusPlus(centroids, numInitializedCentroids);
      }
      double prevSquaredDist = Double.MAX_VALUE;
      int[] centroidSize = new int[centroids.length];
      for (int iter = 0; iter < iters; iter++) {
        squaredDist = runKMeansStep(centroids, centroidSize, vectorCentroids, normalizeCenters);
        // Check for convergence
        if (prevSquaredDist <= (squaredDist + 1e-6)) {
          break;
        }
        Arrays.fill(centroidSize, 0);
        prevSquaredDist = squaredDist;
      }
      if (squaredDist < minSquaredDist) {
        minSquaredDist = squaredDist;
        // Copy out the best centroid as it might be overwritten by the next restart
        bestCentroids = new float[centroids.length][];
        for (int i = 0; i < centroids.length; i++) {
          bestCentroids[i] = ArrayUtil.copyArray(centroids[i]);
        }
      }
    }
    return bestCentroids;
  }

  /**
   * Initialize centroids using Forgy method: randomly select numCentroids vectors for initial
   * centroids
   */
  private void initializeForgy(float[][] initialCentroids, int fromCentroid) throws IOException {
    if (fromCentroid >= numCentroids) {
      return;
    }
    int numCentroids = this.numCentroids - fromCentroid;
    Set<Integer> selection = new HashSet<>();
    while (selection.size() < numCentroids) {
      selection.add(random.nextInt(numVectors));
    }
    int i = 0;
    for (Integer selectedIdx : selection) {
      float[] vector = vectors.vectorValue(selectedIdx);
      System.arraycopy(vector, 0, initialCentroids[fromCentroid + i++], 0, vector.length);
    }
  }

  /** Initialize centroids using a reservoir sampling method */
  private void initializeReservoirSampling(float[][] initialCentroids, int fromCentroid)
      throws IOException {
    if (fromCentroid >= numCentroids) {
      return;
    }
    int numCentroids = this.numCentroids - fromCentroid;
    for (int index = 0; index < numVectors; index++) {
      float[] vector = vectors.vectorValue(index);
      if (index < numCentroids) {
        System.arraycopy(vector, 0, initialCentroids[index + fromCentroid], 0, vector.length);
      } else if (random.nextDouble() < numCentroids * (1.0 / index)) {
        int c = random.nextInt(numCentroids);
        System.arraycopy(vector, 0, initialCentroids[c + fromCentroid], 0, vector.length);
      }
    }
  }

  /** Initialize centroids using Kmeans++ method */
  private void initializePlusPlus(float[][] initialCentroids, int fromCentroid) throws IOException {
    if (fromCentroid >= numCentroids) {
      return;
    }
    // Choose the first centroid uniformly at random
    int firstIndex = random.nextInt(numVectors);
    float[] value = vectors.vectorValue(firstIndex);
    System.arraycopy(value, 0, initialCentroids[fromCentroid], 0, value.length);

    // Store distances of each point to the nearest centroid
    Arrays.fill(kmeansPlusPlusScratch, Float.MAX_VALUE);

    // Step 2 and 3: Select remaining centroids
    for (int i = fromCentroid + 1; i < numCentroids; i++) {
      // Update distances with the new centroid
      double totalSum = 0;
      for (int j = 0; j < numVectors; j++) {
        // TODO: replace with RandomVectorScorer::score possible on quantized vectors
        float dist = VectorUtil.squareDistance(vectors.vectorValue(j), initialCentroids[i - 1]);
        if (dist < kmeansPlusPlusScratch[j]) {
          kmeansPlusPlusScratch[j] = dist;
        }
        totalSum += kmeansPlusPlusScratch[j];
      }

      // Randomly select next centroid
      double r = totalSum * random.nextDouble();
      double cumulativeSum = 0;
      int nextCentroidIndex = 0;
      for (int j = 0; j < numVectors; j++) {
        cumulativeSum += kmeansPlusPlusScratch[j];
        if (cumulativeSum >= r && kmeansPlusPlusScratch[j] > 0) {
          nextCentroidIndex = j;
          break;
        }
      }
      // Update centroid
      value = vectors.vectorValue(nextCentroidIndex);
      System.arraycopy(value, 0, initialCentroids[i], 0, value.length);
    }
  }

  /**
   * Run kmeans step
   *
   * @param centroids centroids, new calculated centroids are written here
   * @param docCentroids for each document which centroid it belongs to, results will be written
   *     here
   * @param normalizeCentroids if centroids should be normalized; used for cosine similarity only
   * @throws IOException if there is an error accessing vector values
   */
  private double runKMeansStep(
      float[][] centroids, int[] centroidSize, short[] docCentroids, boolean normalizeCentroids)
      throws IOException {
    short numCentroids = (short) centroids.length;
    assert Arrays.stream(centroidSize).allMatch(size -> size == 0);
    float[][] newCentroids = new float[numCentroids][centroids[0].length];

    double sumSquaredDist = 0;
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      short bestCentroid = 0;
      if (numCentroids > 1) {
        float minSquaredDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          // TODO: replace with RandomVectorScorer::score possible on quantized vectors
          float squareDist = VectorUtil.squareDistance(centroids[c], vector);
          if (squareDist < minSquaredDist) {
            bestCentroid = c;
            minSquaredDist = squareDist;
          }
        }
        sumSquaredDist += minSquaredDist;
      }

      centroidSize[bestCentroid] += 1;
      for (int dim = 0; dim < vector.length; dim++) {
        newCentroids[bestCentroid][dim] += vector[dim];
      }
      docCentroids[docID] = bestCentroid;
    }

    IntArrayList unassignedCentroids = new IntArrayList();
    for (int c = 0; c < numCentroids; c++) {
      if (centroidSize[c] > 0) {
        for (int dim = 0; dim < newCentroids[c].length; dim++) {
          centroids[c][dim] = newCentroids[c][dim] / centroidSize[c];
        }
      } else {
        unassignedCentroids.add(c);
      }
    }
    if (unassignedCentroids.size() > 0) {
      assignCentroids(vectors, centroids, unassignedCentroids);
    }
    if (normalizeCentroids) {
      for (float[] centroid : centroids) {
        VectorUtil.l2normalize(centroid, false);
      }
    }
    assert Arrays.stream(centroidSize).sum() == vectors.size();
    return sumSquaredDist;
  }

  /**
   * For centroids that did not get any points, assign outlying points to them chose points by
   * descending distance to the current centroid set
   */
  static void assignCentroids(
      FloatVectorValues vectors, float[][] centroids, IntArrayList unassignedCentroidsIdxs)
      throws IOException {
    int[] assignedCentroidsIdxs = new int[centroids.length - unassignedCentroidsIdxs.size()];
    int assignedIndex = 0;
    for (int i = 0; i < centroids.length; i++) {
      if (unassignedCentroidsIdxs.contains(i) == false) {
        assignedCentroidsIdxs[assignedIndex++] = i;
      }
    }
    NeighborQueue queue = new NeighborQueue(unassignedCentroidsIdxs.size(), false);
    for (int i = 0; i < vectors.size(); i++) {
      float[] vector = vectors.vectorValue(i);
      for (short j = 0; j < assignedCentroidsIdxs.length; j++) {
        float squareDist = VectorUtil.squareDistance(centroids[assignedCentroidsIdxs[j]], vector);
        queue.insertWithOverflow(i, squareDist);
      }
    }
    for (int i = 0; i < unassignedCentroidsIdxs.size(); i++) {
      float[] vector = vectors.vectorValue(queue.topNode());
      int unassignedCentroidIdx = unassignedCentroidsIdxs.get(i);
      centroids[unassignedCentroidIdx] = ArrayUtil.copyArray(vector);
      queue.pop();
    }
  }

  /** Kmeans initialization methods */
  public enum KmeansInitializationMethod {
    FORGY,
    RESERVOIR_SAMPLING,
    PLUS_PLUS
  }

  /**
   * Results of KMeans clustering
   *
   * @param centroids the produced centroids
   * @param vectorCentroids for each vector which centroid it belongs to, we use short type, as we
   *     expect less than {@code MAX_NUM_CENTROIDS} which is equal to 32767 centroids. Can be {@code
   *     null} if they were not computed.
   */
  public record Results(float[][] centroids, int[] centroidsSize, short[] vectorCentroids) {}
}
