/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

public final class KMeansLocal {
    private record NeighborInfo(float distanceSq, short offset) implements Comparable<NeighborInfo> {

    @Override
      public int compareTo(NeighborInfo other) {
        return Float.compare(other.distanceSq, this.distanceSq);
      }
    }

  private static void computeNeighborhoods(float[][] centers,
                                           List<short[]> neighborhoods, // Modified in place
                                           int clustersPerNeighborhood) {

    int k = neighborhoods.size();
    if (k == 0 || clustersPerNeighborhood <= 0) {
      return;
    }

    List<PriorityQueue<NeighborInfo>> neighborQueues = new ArrayList<>(k);
    for (int i = 0; i < k; i++) {
      neighborQueues.add(new PriorityQueue<>());
    }

    UpdateNeighborsHelper updateNeighborsHelper = new UpdateNeighborsHelper(clustersPerNeighborhood);

    for (short i = 0; i < k; i++) {
      for (short j = (short) (i+1); j < k; j++) {
          float dsq = VectorUtil.squareDistance(centers[i], centers[j]);
          updateNeighborsHelper.update(i, dsq, neighborQueues.get(i));
          updateNeighborsHelper.update(j, dsq, neighborQueues.get(j));
      }
    }

    for (int i = 0; i < k; i++) {
      PriorityQueue<NeighborInfo> queue = neighborQueues.get(i);
      int neighborCount = queue.size();
      short[] neighbors = new short[neighborCount];
      int idx = 0;
      while (!queue.isEmpty()) {
        neighbors[idx++] = queue.poll().offset;
      }
      Arrays.sort(neighbors);
      neighborhoods.set(i, neighbors);
    }
  }

  private static class UpdateNeighborsHelper {
    private final int maxSize;

    UpdateNeighborsHelper(int clustersPerNeighborhood) {
      this.maxSize = clustersPerNeighborhood;
    }

    void update(short neighborOffset, float distanceSq, PriorityQueue<NeighborInfo> queue) {
      if (queue.size() < maxSize) {
        queue.offer(new NeighborInfo(distanceSq, neighborOffset));
      } else {
        NeighborInfo largestNeighbor = queue.peek();
        if (largestNeighbor != null && distanceSq < largestNeighbor.distanceSq) {
          queue.poll();
          queue.offer(new NeighborInfo(distanceSq, neighborOffset));
        }
      }
    }
  }

  private static boolean stepLloyd(FloatVectorValuesSlice dataset,
                                   List<short[]> neighborhoods,
                                   float[][] centers,
                                   float[][] nextCenters,
                                   long[] centerCounts,
                                   short[] assignments,
                                   float[] assignmentDistances) throws IOException {

    boolean changed = false;
    int dim = centers[0].length;
    int k = centerCounts.length;
    int n = assignments.length;

    Arrays.fill(centerCounts, 0L);
    for(int i = 0; i < nextCenters.length; i++) {
      for(int j = 0; j < nextCenters[0].length; j++) {
        nextCenters[i][j] = 0.0f;
      }
    }

    for (int i = 0; i < n; i++) {
      float[] vector = dataset.vectorValue(i);
      short currentClusterIndex = assignments[i];
      short bestCenterOffset = currentClusterIndex;

      float minDsq = VectorUtil.squareDistance(vector, centers[currentClusterIndex]);

      if (currentClusterIndex < neighborhoods.size()) {
        short[] neighborOffsets = neighborhoods.get(currentClusterIndex);
        if (neighborOffsets != null) {
          for (short neighborOffset : neighborOffsets) {
            if (neighborOffset >= 0 && neighborOffset <= centers.length) {
              float dsq = VectorUtil.squareDistance(vector, centers[neighborOffset]);
              if (dsq < minDsq) {
                minDsq = dsq;
                bestCenterOffset = neighborOffset;
              }
            }
          }
        }
      }
      if (assignments[i] != bestCenterOffset) {
        changed = true;
      }
      assignments[i] = bestCenterOffset;
      assignmentDistances[i] = minDsq;

      if (bestCenterOffset >= 0 && bestCenterOffset <= centers.length) {
        centerCounts[bestCenterOffset]++;
        for (short d = 0; d < dim; d++) {
          nextCenters[bestCenterOffset][d] += vector[d];
        }
      }
    }

    for (int clusterIdx = 0; clusterIdx < k; clusterIdx++) {
      if (centerCounts[clusterIdx] > 0) {
        float countF = (float) centerCounts[clusterIdx];
        for (int d = 0; d < dim; d++) {
          centers[clusterIdx][d] = nextCenters[clusterIdx][d] / countF;
        }
      }
    }

    return changed;
  }

  static void assignSpilled(FloatVectorValuesSlice vectors, List<short[]> neighborhoods, float[][] centers, short[] assignments, short[] spilledAssignments, float[] spilledDistances) throws IOException {
    // SOAR uses an adjusted distance for assigning spilled documents which is
    // given by:
    //
    //   soar(x, c) = ||x - c||^2 + lambda * ((x - c_1)^t (x - c))^2 / ||x - c_1||^2
    //
    // Here, x is the document, c is the nearest centroid, and c_1 is the first
    // centroid the document was assigned to. The document is assigned to the
    // cluster with the smallest soar(x, c).

    float[] d1 = new float[vectors.dimension()];
    for(int i = 0; i < vectors.size(); i++) {
      float[] xi = vectors.vectorValue(i);

      short currJd = assignments[i];
      float[] c1 = centers[currJd];
      float d1sq = VectorUtil.squareDistance(xi, c1);

      short bestJd = 0;
      float minSoar = Float.MAX_VALUE;
      for(short jd : neighborhoods.get(currJd)) {
        float[] cj = centers[jd];
        float soar = distanceSoar(d1, xi, cj, d1sq);
        if(soar < minSoar) {
          bestJd = jd;
          minSoar = soar;
        }
      }

      spilledAssignments[i] = bestJd;
      spilledDistances[i] = minSoar;
    }
  }

  static float distanceSoar(float[] r, float[] x, float[] c, float rnorm) {
    float lambda = 1.0F;

    float dsq = 0.0f;
    float rproj = 0.0f;

    for(int d = 0; d < x.length; d++) {
      float diff = x[d] - c[d];
      dsq += diff * diff;
      rproj += r[d] * diff;
    }

    return dsq * lambda * rproj * rproj / rnorm;
  }

  public static DefaultIVFVectorsWriter.KMeansResult kMeansLocal(FloatVectorValuesSlice dataset,
                                         final float[][] centers,
                                         final short[] assignments,
                                         final int[] assignmentOrds,
                                         final float[] assignmentDistances,
                                         short clustersPerNeighborhood,
                                         int maxIterations) throws IOException {
    int k = centers.length;

    List<short[]> neighborhoods = new ArrayList<>(k);
    for(int i=0; i < k; ++i) {
      neighborhoods.add(null);
    }

    computeNeighborhoods(centers, neighborhoods, clustersPerNeighborhood);

    boolean converged = false;
    long[] centerCounts = new long[k];
    float[][] nextCenters = new float[centers.length][centers[0].length];

    int iterationsRun;
    for (iterationsRun = 0; iterationsRun < maxIterations; iterationsRun++) {
      boolean changed = stepLloyd(dataset, neighborhoods, centers, nextCenters, centerCounts, assignments, assignmentDistances);
      if (!changed) {
        converged = true;
        break;
      }
    }

    short[] spilledAssignments = new short[assignments.length];
    int[] spilledAssignmentOrds = new int[assignmentOrds.length];
    float[] spilledDistances = new float[assignments.length];

    long startTime = System.nanoTime();

    assignSpilled(dataset, neighborhoods, centers, assignments, spilledAssignments, spilledDistances);

    System.out.println(" ==== assign soar ms: " + (System.nanoTime() - startTime) / 1000000.0);

    return new DefaultIVFVectorsWriter.KMeansResult(centers,
      assignments, assignmentOrds, assignmentDistances,
      spilledAssignments, spilledAssignmentOrds, spilledDistances,
      iterationsRun, converged);
  }
}