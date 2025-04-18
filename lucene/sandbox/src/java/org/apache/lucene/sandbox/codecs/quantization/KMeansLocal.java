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
    private record NeighborInfo(float distanceSq, long offset) implements Comparable<NeighborInfo> {

    @Override
      public int compareTo(NeighborInfo other) {
        // Reverse order for max-heap behavior based on distance
        return Float.compare(other.distanceSq, this.distanceSq);
      }
    }

  private static void computeNeighborhoods(float[][] centers,
                                           List<long[]> neighborhoods, // Modified in place
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

    // Compute pairwise distances and update neighborhood queues
    for (int i = 0; i < k; i++) {
      for (int j = i+1; j < k; j++) {
          float dsq = VectorUtil.squareDistance(centers[i], centers[j]);
          updateNeighborsHelper.update(i, dsq, neighborQueues.get(i));
          updateNeighborsHelper.update(j, dsq, neighborQueues.get(j));
      }
    }

    // Extract neighbor offsets from queues and sort them
    for (int i = 0; i < k; i++) {
      PriorityQueue<NeighborInfo> queue = neighborQueues.get(i);
      int neighborCount = queue.size();
      long[] neighbors = new long[neighborCount];
      int idx = 0;
      while (!queue.isEmpty()) {
        // Store the offset of the neighbor
        neighbors[idx++] = queue.poll().offset;
      }
      // Sort neighbors by offset (index in the centers array)
      Arrays.sort(neighbors);
      neighborhoods.set(i, neighbors);
    }
  }

  private static class UpdateNeighborsHelper {
    private final int maxSize;

    UpdateNeighborsHelper(int clustersPerNeighborhood) {
      this.maxSize = clustersPerNeighborhood;
    }

    void update(long neighborOffset, float distanceSq, PriorityQueue<NeighborInfo> queue) {
      if (queue.size() < maxSize) {
        queue.offer(new NeighborInfo(distanceSq, neighborOffset));
      } else {
        // Queue is full, check if new distance is smaller than the largest distance currently in the queue
        NeighborInfo largestNeighbor = queue.peek(); // Peek returns element with highest priority (max distance)
        if (largestNeighbor != null && distanceSq < largestNeighbor.distanceSq) {
          queue.poll(); // Remove the neighbor with the largest distance
          queue.offer(new NeighborInfo(distanceSq, neighborOffset)); // Add the new, closer neighbor
        }
      }
    }
  }

  private static boolean stepLloyd(FloatVectorValuesSlice dataset,
                                   List<long[]> neighborhoods, // Use precomputed neighborhoods
                                   float[][] centers,        // Modifies this in-place
                                   float[][] nextCenters,    // Used as temp buffer
                                   long[] q,               // Used as temp buffer (counts)
                                   short[] assignments) throws IOException {

    boolean changed = false;
    int dim = centers[0].length;
    int k = q.length; // Number of clusters
    int n = assignments.length; // Number of data points

    // Reset buffers for the current iteration
    Arrays.fill(q, 0L);
    for(int i = 0; i < nextCenters.length; i++) {
      for(int j = 0; j < nextCenters[0].length; j++) {
        nextCenters[i][j] = 0.0f;
      }
    }

//    System.out.println(" ==== c len: " + centers.length);
//    System.out.println(" ==== assignments: " + Arrays.toString(assignments));

    for (int i = 0; i < n; i++) {
      float[] vector = dataset.vectorValue(i);
      short currentClusterIndex = assignments[i];
      long bestCenterOffset = currentClusterIndex; // Start assuming current center is best

//      System.out.println(" ==== current cluster index: " + currentClusterIndex);

      // Calculate distance to the *currently assigned* center first
      float minDsq = VectorUtil.squareDistance(vector, centers[currentClusterIndex]);

      // Check neighborhood of the *current* cluster
      if (currentClusterIndex < neighborhoods.size()) {
        long[] neighborOffsets = neighborhoods.get(currentClusterIndex);
        if (neighborOffsets != null) {
          for (long neighborOffset : neighborOffsets) {
            // Ensure neighbor offset is valid before calculating distance
            if (neighborOffset >= 0 && neighborOffset <= centers.length) {
              float dsq = VectorUtil.squareDistance(vector, centers[(int)neighborOffset]);
              if (dsq < minDsq) {
                minDsq = dsq;
                bestCenterOffset = neighborOffset;
              }
            }
          }
        }
      }
      // Check if assignment changed
      if (assignments[i] != bestCenterOffset) {
        changed = true;
      }
      assignments[i] = (short) bestCenterOffset; // Update assignment (store offset)

      // Update count and sum for the (potentially newly) assigned cluster
      // Ensure bestCenterOffset is valid before using it
      if (bestCenterOffset >= 0 && bestCenterOffset <= centers.length) {
        q[(int)bestCenterOffset]++;
        for (int d = 0; d < dim; d++) {
          nextCenters[(int)bestCenterOffset][d] += vector[d];
        }
      }
    }

    // --- Update Step (Identical to original stepLloyd) ---
    // Iterate through each cluster and update its center
    for (int clusterIdx = 0; clusterIdx < k; clusterIdx++) {
      if (q[clusterIdx] > 0) {
        float countF = (float) q[clusterIdx];
        // Calculate new center by dividing sum by count
        for (int d = 0; d < dim; d++) {
          centers[clusterIdx][d] = nextCenters[clusterIdx][d] / countF;
        }
      }
    }

    return changed;
  }

  public static DefaultIVFVectorsWriter.KMeansResult kMeansLocal(FloatVectorValuesSlice dataset,
                                         final float[][] initialCenters,
                                         final short[] initialAssignments,
                                         final int[] assignmentOrdinals,
                                         int clustersPerNeighborhood,
                                         int maxIterations) throws IOException {

    // FIXME: remove garbage ai commentary
    // FIXME: add back input validation?
    // FIXME: optimize this code
    // FIXME: don't make copies??

    int k = initialCenters.length;
    int n = dataset.size();

//    float[][] centers = Arrays.copyOf(initialCenters, initialCenters.length);
//    short[] assignments = Arrays.copyOf(initialAssignments, initialAssignments.length);

    float[][] centers = initialCenters;
    short[] assignments = initialAssignments;

//    System.out.println(" ==== assignments len: " + assignments.length);
//    System.out.println(" ==== assignmentsOrds len: " + assignmentOrdinals.length);

    if (k == 1 || k >= n) {
      // No iterations needed, return the initial state (copied)
      boolean converged = true; // Already in final state
      return new DefaultIVFVectorsWriter.KMeansResult(centers, assignments, assignmentOrdinals, 0, converged);
    }
    // --- Compute Neighborhoods ---
    List<long[]> neighborhoods = new ArrayList<>(k);
    // Initialize the list structure before passing it
    for(int i=0; i < k; ++i) {
      neighborhoods.add(null);
    }

//    System.out.println(" === compute neighborhoods ");

    computeNeighborhoods(centers, neighborhoods, clustersPerNeighborhood);

    int iterationsRun = 0;
    boolean converged = false;
    long[] q = new long[k];            // FIXME: rename this? ... Buffer for counts
    float[][] nextCenters = new float[centers.length][centers[0].length]; // FIXME: rename this ... Buffer for sums

    for (iterationsRun = 0; iterationsRun < maxIterations; iterationsRun++) {

//      System.out.println(" === lloyd ");

      // Use the neighborhood-aware stepLloyd
      boolean changed = stepLloyd(dataset, neighborhoods, centers, nextCenters, q, assignments);
      if (!changed) {
        converged = true;
        break;
      }
    }

//    System.out.println(" ==== assignments len 2: " + assignments.length);
//    System.out.println(" ==== assignmentsOrds len 2: " + assignmentOrdinals.length);

    // Create the result object - constructor takes ownership of the modified copies
    return new DefaultIVFVectorsWriter.KMeansResult(centers, assignments, assignmentOrdinals, iterationsRun, converged);
  }
}