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
 */

package org.apache.lucene.util.hnsw;

import org.apache.lucene.util.LongHeap;
import org.apache.lucene.util.NumericUtils;

import java.util.HashMap;

/**
 * NeighborQueue uses a {@link LongHeap} to store lists of arcs in an HNSW graph, represented as a
 * neighbor node id with an associated score packed together as a sortable long, which is sorted
 * primarily by score. The queue provides both fixed-size and unbounded operations via {@link
 * #insertWithOverflow(int, float)} and {@link #add(int, float)}, and provides MIN and MAX heap
 * subclasses.
 */
public class NeighborQueue {

  private enum Order {
    NATURAL {
      @Override
      long apply(long v) {
        return v;
      }
    },
    REVERSED {
      @Override
      long apply(long v) {
        // This cannot be just `-v` since Long.MIN_VALUE doesn't have a positive counterpart. It
        // needs a function that returns MAX_VALUE for MIN_VALUE and vice-versa.
        return -1 - v;
      }
    };

    abstract long apply(long v);
  }

  private final LongHeap heap;
  private final HashMap<Integer,Integer> nodeIdToHeapIndex;
  private final Order order;

  // Used to track the number of neighbors visited during a single graph traversal
  private int visitedCount;
  // Whether the search stopped early because it reached the visited nodes limit
  private boolean incomplete;

  public NeighborQueue(int initialSize, boolean reversed) {
    this.heap = new LongHeap(initialSize);
    this.order = reversed ? Order.REVERSED : Order.NATURAL;
    this.nodeIdToHeapIndex = new HashMap<>(initialSize);
  }

  /** @return the number of elements in the heap */
  public int size() {
    return heap.size();
  }

  /**
   * Adds a new graph arc, extending the storage as needed.
   *
   * @param nodeId the neighbor node id
   * @param nodeScore the score of the neighbor, relative to some other node
   */
  public void add(int nodeId, float nodeScore) {
    heap.push(encode(nodeId, nodeScore));
  }
  
  /**
   * Adds a new graph arc, extending the storage as needed.
   * This variant is more expensive but it is compatible with a multi-valued scenario.
   *
   * @param nodeId the neighbor node id
   * @param nodeScore the score of the neighbor, relative to some other node
   */
  public void add(int nodeId, float nodeScore, HnswGraphSearcher.Multivalued strategy) {
    if(strategy.equals(HnswGraphSearcher.Multivalued.NONE)){
      this.add(nodeId,nodeScore);
    } else {
      Integer heapIndex = nodeIdToHeapIndex.get(nodeId);
      if (heapIndex == null) {
        heapIndex = heap.push(encode(nodeId, nodeScore));
      } else {
        float originalScore = decodeScore(heap.get(heapIndex));
        float updatedScore = strategy.updateScore(originalScore, nodeScore);
        heapIndex = heap.updateElement(heapIndex, encode(nodeId, updatedScore));
      }
      nodeIdToHeapIndex.put(nodeId, heapIndex);
    }
  }

  /**
   * If the heap is not full (size is less than the initialSize provided to the constructor), adds a
   * new node-and-score element. If the heap is full, compares the score against the current top
   * score, and replaces the top element if newScore is better than (greater than unless the heap is
   * reversed), the current top score.
   *
   * @param nodeId the neighbor node id
   * @param nodeScore the score of the neighbor, relative to some other node
   */
  public boolean insertWithOverflow(int nodeId, float nodeScore) {
    return (heap.insertWithOverflow(encode(nodeId, nodeScore)) != -1);
  }
  
  /**
   * If the heap is not full (size is less than the initialSize provided to the constructor), adds a
   * new node-and-score element. If the heap is full, compares the score against the current top
   * score, and replaces the top element if newScore is better than (greater than unless the heap is
   * reversed), the current top score.
   *
   * @param nodeId the neighbor node id
   * @param nodeScore the score of the neighbor, relative to some other node
   */
  public boolean insertWithOverflow(int nodeId, float nodeScore, HnswGraphSearcher.Multivalued strategy) {
    if (strategy.equals(HnswGraphSearcher.Multivalued.NONE)) {
      return insertWithOverflow(nodeId, nodeScore);
    } else {
      boolean nodeAdded = false;
      Integer heapIndex = nodeIdToHeapIndex.get(nodeId);
      if (heapIndex == null) {
        int minNodeId = this.topNode();
        heapIndex = heap.insertWithOverflow(encode(nodeId, nodeScore));
        if (heapIndex != -1) {
          this.nodeIdToHeapIndex.remove(minNodeId);
          nodeAdded = true;
        }
      } else {
        float originalScore = decodeScore(heap.get(heapIndex));
        float updatedScore = strategy.updateScore(originalScore, nodeScore);
        heapIndex = heap.updateElement(heapIndex, encode(nodeId, updatedScore));
      }
      nodeIdToHeapIndex.put(nodeId, heapIndex);

      return nodeAdded;
    }
  }

  private long encode(int nodeId, float score) {
    return order.apply((((long) NumericUtils.floatToSortableInt(score)) << 32) | nodeId);
  }

  private float decodeScore(long heapValue) {
    return NumericUtils.sortableIntToFloat((int) (order.apply(heapValue) >> 32));
  }

  private int decodeNodeId(long heapValue) {
    return (int) order.apply(heapValue);
  }

  /** Removes the top element and returns its node id. */
  public int pop() {
    return decodeNodeId(heap.pop());
  }

  public int[] nodes() {
    int size = size();
    int[] nodes = new int[size];
    for (int i = 0; i < size; i++) {
      nodes[i] = (int) order.apply(heap.get(i + 1));
    }
    return nodes;
  }

  /** Returns the top element's node id. */
  public int topNode() {
    return decodeNodeId(heap.top());
  }

  /** Returns the top element's node score. */
  public float topScore() {
    return decodeScore(heap.top());
  }

  public void clear() {
    heap.clear();
    visitedCount = 0;
  }

  public int visitedCount() {
    return visitedCount;
  }

  public void setVisitedCount(int visitedCount) {
    this.visitedCount = visitedCount;
  }

  public boolean incomplete() {
    return incomplete;
  }

  public void markIncomplete() {
    this.incomplete = true;
  }

  @Override
  public String toString() {
    return "Neighbors[" + heap.size() + "]";
  }
}
