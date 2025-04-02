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
 *
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
import org.apache.lucene.internal.hppc.IntObjectHashMap;
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
      centroids = kmeans.computeCentroids();
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
  private final float[] lowerBounds;
  private final float[] upperBounds;
  private final float[] s;
  private final float[] centerMovement;
  private final float[][] sumNewCenters;

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
    this.lowerBounds = new float[vectors.size()];
    this.upperBounds = new float[vectors.size()];
    Arrays.fill(upperBounds, Float.MAX_VALUE);
    this.s = new float[numCentroids];
    this.centerMovement = new float[numCentroids];
    this.sumNewCenters = new float[numCentroids][vectors.dimension()];
  }

  private void initS(float[][] centroids) {
    for (int i = 0; i < centroids.length; i++) {
      float[] c1 = centroids[i];
      s[i] = Float.MAX_VALUE;
      for (int j = 0; j < centroids.length; j++) {
        if (i == j) {
          continue;
        }
        float[] c2 = centroids[j];
        float dist = VectorUtil.squareDistance(c1, c2);
        if (dist < s[i]) {
          s[i] = dist;
        }
      }
      s[i] = (float) Math.sqrt(s[i]) / 2f;
    }
  }

  private void changeAssignment(
      int vecId, short newCluster, int[] centroidSizes, short[] docCentroids) throws IOException {
    VectorUtil.subtract(
        sumNewCenters[docCentroids[vecId]],
        vectors.vectorValue(vecId),
        sumNewCenters[docCentroids[vecId]]);
    VectorUtil.add(sumNewCenters[newCluster], vectors.vectorValue(vecId));
    centroidSizes[docCentroids[vecId]]--;
    centroidSizes[newCluster]++;
    docCentroids[vecId] = newCluster;
  }

  private void assign(
      float[][] centroids, int[] centroidSize, short[] docCentroids, float[] upperBounds)
      throws IOException {
    for (int docID = 0; docID < vectors.size(); docID++) {
      float[] vector = vectors.vectorValue(docID);
      short bestCentroid = 0;
      if (numCentroids > 1) {
        float minSquaredDist = Float.MAX_VALUE;
        for (short c = 0; c < numCentroids; c++) {
          float squareDist = VectorUtil.squareDistance(centroids[c], vector);
          if (squareDist < minSquaredDist) {
            bestCentroid = c;
            minSquaredDist = squareDist;
          }
        }
        upperBounds[docID] = (float) Math.sqrt(minSquaredDist);
      }

      centroidSize[bestCentroid] += 1;
      docCentroids[docID] = bestCentroid;
      VectorUtil.add(sumNewCenters[bestCentroid], vector);
    }
    IntArrayList unassignedCentroids = new IntArrayList();
    for (int c = 0; c < numCentroids; c++) {
      if (centroidSize[c] == 0) {
        unassignedCentroids.add(c);
      }
    }
    if (unassignedCentroids.size() > 0) {
      throwAwayAndSplitCentroids2(
          centroids, docCentroids, centroidSize, upperBounds, unassignedCentroids);
    }
    moveCenters(centroids, sumNewCenters, centroidSize);
  }

  private int[] moveCenters(float[][] centroids, float[][] sumNewCenters, int[] centroidSize) {
    int furtherestMovingCenter = -1;
    int secondFurtherestMovingCenter = -1;
    float[] newCenter = new float[centroids[0].length];
    for (int i = 0; i < centroids.length; i++) {
      centerMovement[i] = 0;
      if (centroidSize[i] == 0) {
        continue;
      }
      float[] center = centroids[i];
      System.arraycopy(sumNewCenters[i], 0, newCenter, 0, newCenter.length);
      int size = centroidSize[i];
      for (int dim = 0; dim < center.length; dim++) {
        newCenter[dim] /= size;
        centerMovement[i] += (center[dim] - newCenter[dim]) * (center[dim] - newCenter[dim]);
        center[dim] = newCenter[dim];
      }
      centerMovement[i] = (float) Math.sqrt(centerMovement[i]);
      if (furtherestMovingCenter == -1) {
        furtherestMovingCenter = i;
      } else if (centerMovement[i] > centerMovement[furtherestMovingCenter]) {
        secondFurtherestMovingCenter = furtherestMovingCenter;
        furtherestMovingCenter = i;
      } else if (secondFurtherestMovingCenter == -1
          || centerMovement[i] > centerMovement[secondFurtherestMovingCenter]) {
        secondFurtherestMovingCenter = i;
      }
    }
    return new int[] {furtherestMovingCenter, secondFurtherestMovingCenter};
  }

  private void updateBounds(short[] docCentroids, int[] furthestMoving) {
    for (int i = 0; i < docCentroids.length; i++) {
      upperBounds[i] += centerMovement[docCentroids[i]];
      lowerBounds[i] -=
          (docCentroids[i] == furthestMoving[0]
              ? centerMovement[furthestMoving[1]]
              : centerMovement[furthestMoving[0]]);
    }
  }

  public float[][] computeCentroids() throws IOException {
    short[] vectorCentroids = new short[numVectors];
    float[][] centroids = new float[numCentroids][vectors.dimension()];
    int numInitializedCentroids = 0;
    // The user has given us a solid number of centroids to start of with, so skip restarts, fill in
    // where we can, and refine
    if (initCentroids != null && initCentroids.length > numCentroids / 2) {
      int i = 0;
      for (; i < Math.min(numCentroids, initCentroids.length); i++) {
        System.arraycopy(initCentroids[i], 0, centroids[i], 0, initCentroids[i].length);
      }
      numInitializedCentroids = i;
    }

    switch (initializationMethod) {
      case FORGY -> initializeForgy(centroids, numInitializedCentroids);
      case RESERVOIR_SAMPLING -> initializeReservoirSampling(centroids, numInitializedCentroids);
      case PLUS_PLUS -> initializePlusPlus(centroids, numInitializedCentroids);
    }
    int[] centroidSize = new int[centroids.length];
    assign(centroids, centroidSize, vectorCentroids, upperBounds);
    for (int iter = 0; iter < iters; iter++) {
      initS(centroids);
      boolean changed = false;
      for (int i = 0; i < vectors.size(); i++) {
        float[] vector = vectors.vectorValue(i);
        short nearest = vectorCentroids[i];
        float upperComparisonBound = Math.max(s[nearest], lowerBounds[i]);
        if (upperBounds[i] <= upperComparisonBound) {
          continue;
        }
        float newLower = Float.MAX_VALUE;
        float u2 = upperBounds[i] * upperBounds[i];
        for (int j = 0; j < numCentroids; j++) {
          if (j == nearest) {
            continue;
          }
          float dist = VectorUtil.squareDistance(centroids[j], vector);
          if (dist < u2) {
            newLower = u2;
            u2 = dist;
            nearest = (short) j;
          } else if (dist < newLower) {
            newLower = dist;
          }
        }
        lowerBounds[i] = (float) Math.sqrt(newLower);
        if (nearest != vectorCentroids[i]) {
          changed = true;
          upperBounds[i] = (float) Math.sqrt(u2);
          changeAssignment(i, nearest, centroidSize, vectorCentroids);
        }
      }
      if (changed == false) {
        break;
      }
      int[] furthestMovingCenter = moveCenters(centroids, sumNewCenters, centroidSize);
      // if we moved tiny, we are converged
      if (centerMovement[furthestMovingCenter[0]] <= 1e-6) {
        break;
      }
      updateBounds(vectorCentroids, furthestMovingCenter);
    }
    return centroids;
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
      throwAwayAndSplitCentroids(
          vectors, centroids, docCentroids, centroidSize, unassignedCentroids);
    }
    if (normalizeCentroids) {
      for (float[] centroid : centroids) {
        VectorUtil.l2normalize(centroid, false);
      }
    }
    assert Arrays.stream(centroidSize).sum() == vectors.size();
    return sumSquaredDist;
  }

  void throwAwayAndSplitCentroids2(
      float[][] centroids,
      short[] docCentroids,
      int[] centroidSize,
      float[] upperBounds,
      IntArrayList unassignedCentroidsIdxs)
      throws IOException {
    IntObjectHashMap<IntArrayList> splitCentroids =
        new IntObjectHashMap<>(unassignedCentroidsIdxs.size());
    // used for splitting logic
    int[] splitSizes = Arrays.copyOf(centroidSize, centroidSize.length);
    // FAISS style algorithm for splitting
    for (int i = 0; i < unassignedCentroidsIdxs.size(); i++) {
      int toSplit;
      for (toSplit = 0; true; toSplit = (toSplit + 1) % centroidSize.length) {
        /* probability to pick this cluster for split */
        double p =
            (splitSizes[toSplit] - 1.0) / (float) (docCentroids.length - centroidSize.length);
        float r = random.nextFloat();
        if (r < p) {
          break; /* found our cluster to be split */
        }
      }
      int unassignedCentroidIdx = unassignedCentroidsIdxs.get(i);
      // keep track of those that are split, this way we reassign docCentroids and fix up true size
      // & centroids
      splitCentroids.getOrDefault(toSplit, new IntArrayList()).add(unassignedCentroidIdx);
      System.arraycopy(
          centroids[toSplit],
          0,
          centroids[unassignedCentroidIdx],
          0,
          centroids[unassignedCentroidIdx].length);
      for (int dim = 0; dim < centroids[unassignedCentroidIdx].length; dim++) {
        if (dim % 2 == 0) {
          centroids[unassignedCentroidIdx][dim] *= (1 + EPS);
          centroids[toSplit][dim] *= (1 - EPS);
        } else {
          centroids[unassignedCentroidIdx][dim] *= (1 - EPS);
          centroids[toSplit][dim] *= (1 + EPS);
        }
      }
      splitSizes[unassignedCentroidIdx] = splitSizes[toSplit] / 2;
      splitSizes[toSplit] -= splitSizes[unassignedCentroidIdx];
    }
    // now we need to reassign docCentroids and fix up true size & centroids
    for (int i = 0; i < docCentroids.length; i++) {
      int docCentroid = docCentroids[i];
      IntArrayList split = splitCentroids.get(docCentroid);
      if (split != null) {
        // we need to reassign this doc
        int bestCentroid = docCentroid;
        float bestDist = VectorUtil.squareDistance(centroids[docCentroid], vectors.vectorValue(i));
        for (int j = 0; j < split.size(); j++) {
          int newCentroid = split.get(j);
          float dist = VectorUtil.squareDistance(centroids[newCentroid], vectors.vectorValue(i));
          if (dist < bestDist) {
            bestCentroid = newCentroid;
            bestDist = dist;
          }
        }
        if (bestCentroid != docCentroid) {
          changeAssignment(i, (short) bestCentroid, centroidSize, docCentroids);
          upperBounds[i] = (float) Math.sqrt(bestDist);
        }
      }
    }
  }

  void throwAwayAndSplitCentroids(
      FloatVectorValues vectors,
      float[][] centroids,
      short[] docCentroids,
      int[] centroidSize,
      IntArrayList unassignedCentroidsIdxs)
      throws IOException {
    IntObjectHashMap<IntArrayList> splitCentroids =
        new IntObjectHashMap<>(unassignedCentroidsIdxs.size());
    // used for splitting logic
    int[] splitSizes = Arrays.copyOf(centroidSize, centroidSize.length);
    // FAISS style algorithm for splitting
    for (int i = 0; i < unassignedCentroidsIdxs.size(); i++) {
      int toSplit;
      for (toSplit = 0; true; toSplit = (toSplit + 1) % centroids.length) {
        /* probability to pick this cluster for split */
        double p = (splitSizes[toSplit] - 1.0) / (float) (docCentroids.length - centroids.length);
        float r = random.nextFloat();
        if (r < p) {
          break; /* found our cluster to be split */
        }
      }
      int unassignedCentroidIdx = unassignedCentroidsIdxs.get(i);
      // keep track of those that are split, this way we reassign docCentroids and fix up true size
      // & centroids
      splitCentroids.getOrDefault(toSplit, new IntArrayList()).add(unassignedCentroidIdx);
      System.arraycopy(
          centroids[toSplit],
          0,
          centroids[unassignedCentroidIdx],
          0,
          centroids[unassignedCentroidIdx].length);
      for (int dim = 0; dim < centroids[unassignedCentroidIdx].length; dim++) {
        if (dim % 2 == 0) {
          centroids[unassignedCentroidIdx][dim] *= (1 + EPS);
          centroids[toSplit][dim] *= (1 - EPS);
        } else {
          centroids[unassignedCentroidIdx][dim] *= (1 - EPS);
          centroids[toSplit][dim] *= (1 + EPS);
        }
      }
      splitSizes[unassignedCentroidIdx] = splitSizes[toSplit] / 2;
      splitSizes[toSplit] -= splitSizes[unassignedCentroidIdx];
    }
    // now we need to reassign docCentroids and fix up true size & centroids
    for (int i = 0; i < docCentroids.length; i++) {
      int docCentroid = docCentroids[i];
      IntArrayList split = splitCentroids.get(docCentroid);
      if (split != null) {
        // we need to reassign this doc
        int bestCentroid = docCentroid;
        float bestDist = VectorUtil.squareDistance(centroids[docCentroid], vectors.vectorValue(i));
        for (int j = 0; j < split.size(); j++) {
          int newCentroid = split.get(j);
          float dist = VectorUtil.squareDistance(centroids[newCentroid], vectors.vectorValue(i));
          if (dist < bestDist) {
            bestCentroid = newCentroid;
            bestDist = dist;
          }
        }
        if (bestCentroid != docCentroid) {
          // we need to update the centroid size
          centroidSize[docCentroid]--;
          centroidSize[bestCentroid]++;
          docCentroids[i] = (short) bestCentroid;
          // we need to update the old and new centroid accounting for size as well
          for (int dim = 0; dim < centroids[docCentroid].length; dim++) {
            centroids[docCentroid][dim] -= vectors.vectorValue(i)[dim] / centroidSize[docCentroid];
            centroids[bestCentroid][dim] +=
                vectors.vectorValue(i)[dim] / centroidSize[bestCentroid];
          }
        }
      }
    }
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
