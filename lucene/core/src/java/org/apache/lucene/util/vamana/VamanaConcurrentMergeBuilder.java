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
package org.apache.lucene.util.vamana;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/**
 * A graph builder that manages multiple workers, it only supports adding the whole graph all at
 * once. It will spawn a thread for each worker and the workers will pick the work in batches.
 */
public class VamanaConcurrentMergeBuilder implements VamanaBuilder {

  private static final int DEFAULT_BATCH_SIZE =
      2048; // number of vectors the worker handles sequentially at one batch

  private final TaskExecutor exec;
  private final ConcurrentMergeWorker[] workers;
  private InfoStream infoStream = InfoStream.getDefault();

  public VamanaConcurrentMergeBuilder(
      TaskExecutor exec,
      int numWorker,
      RandomVectorScorerSupplier scorerSupplier,
      int M,
      int beamWidth,
      float alpha,
      OnHeapVamanaGraph vamana,
      BitSet initializedNodes)
      throws IOException {
    this.exec = exec;
    AtomicInteger workProgress = new AtomicInteger(0);
    workers = new ConcurrentMergeWorker[numWorker];
    for (int i = 0; i < numWorker; i++) {
      workers[i] =
          new ConcurrentMergeWorker(
              scorerSupplier.copy(), M, beamWidth, alpha, vamana, initializedNodes, workProgress);
    }
  }

  @Override
  public OnHeapVamanaGraph build(int maxOrd) throws IOException {
    if (infoStream.isEnabled(VamanaGraphBuilder.VAMANA_COMPONENT)) {
      infoStream.message(
          VamanaGraphBuilder.VAMANA_COMPONENT,
          "build graph from " + maxOrd + " vectors, with " + workers.length + " workers");
    }
    List<Callable<Void>> tasks = new ArrayList<>(workers.length);
    for (int i = 0; i < workers.length; i++) {
      int finalI = i;
      tasks.add(
          () -> {
            try {
              workers[finalI].run(maxOrd);
            } catch (IOException e) {
              throw new RuntimeException(e);
            }
            return null;
          });
    }
    exec.invokeAll(tasks);
    return workers[0].getGraph();
  }

  @Override
  public void addGraphNode(int node) throws IOException {
    throw new UnsupportedOperationException("This builder is for merge only");
  }

  @Override
  public void finish() throws IOException {
    workers[0].finish();
  }

  @Override
  public void setInfoStream(InfoStream infoStream) {
    this.infoStream = infoStream;
    for (VamanaBuilder worker : workers) {
      worker.setInfoStream(infoStream);
    }
  }

  @Override
  public OnHeapVamanaGraph getGraph() {
    return workers[0].getGraph();
  }

  /* test only for now */
  void setBatchSize(int newSize) {
    for (ConcurrentMergeWorker worker : workers) {
      worker.batchSize = newSize;
    }
  }

  private static final class ConcurrentMergeWorker extends VamanaGraphBuilder {

    /**
     * A common AtomicInteger shared among all workers, used for tracking what's the next vector to
     * be added to the graph.
     */
    private final AtomicInteger workProgress;

    private final BitSet initializedNodes;
    private int batchSize = DEFAULT_BATCH_SIZE;

    private ConcurrentMergeWorker(
        RandomVectorScorerSupplier scorerSupplier,
        int M,
        int beamWidth,
        float alpha,
        OnHeapVamanaGraph vamana,
        BitSet initializedNodes,
        AtomicInteger workProgress)
        throws IOException {
      super(
          scorerSupplier,
          M,
          beamWidth,
          alpha,
          vamana,
          new MergeSearcher(
              new NeighborQueue(beamWidth, true), new FixedBitSet(vamana.maxNodeId() + 1)));
      this.workProgress = workProgress;
      this.initializedNodes = initializedNodes;
    }

    /**
     * This method first try to "reserve" part of work by calling {@link #getStartPos(int)} and then
     * calling {@link #addVectors(int, int)} to actually add the nodes to the graph. By doing this
     * we are able to dynamically allocate the work to multiple workers and try to make all of them
     * finishing around the same time.
     */
    private void run(int maxOrd) throws IOException {
      int start = getStartPos(maxOrd);
      int end;
      while (start != -1) {
        end = Math.min(maxOrd, start + batchSize);
        addVectors(start, end);
        start = getStartPos(maxOrd);
      }
    }

    /** Reserve the work by atomically increment the {@link #workProgress} */
    private int getStartPos(int maxOrd) {
      int start = workProgress.getAndAdd(batchSize);
      if (start < maxOrd) {
        return start;
      } else {
        return -1;
      }
    }

    @Override
    public void addGraphNode(int node) throws IOException {
      if (initializedNodes != null && initializedNodes.get(node)) {
        return;
      }
      super.addGraphNode(node);
    }
  }

  /**
   * This searcher will obtain the lock and make a copy of neighborArray when seeking the graph such
   * that concurrent modification of the graph will not impact the search
   */
  private static class MergeSearcher extends VamanaGraphSearcher {

    private int[] nodeBuffer;
    private int upto;
    private int size;

    private MergeSearcher(NeighborQueue candidates, BitSet visited) {
      super(candidates, visited);
    }

    @Override
    void graphSeek(VamanaGraph graph, int targetNode) {
      NeighborArray neighborArray = ((OnHeapVamanaGraph) graph).getNeighbors(targetNode);
      neighborArray.rwlock.readLock().lock();
      try {
        if (nodeBuffer == null || nodeBuffer.length < neighborArray.size()) {
          nodeBuffer = new int[neighborArray.size()];
        }
        size = neighborArray.size();
        if (size >= 0) {
          System.arraycopy(neighborArray.nodes, 0, nodeBuffer, 0, size);
        }
      } finally {
        neighborArray.rwlock.readLock().unlock();
      }
      upto = -1;
    }

    @Override
    int graphNextNeighbor(VamanaGraph graph) {
      if (++upto < size) {
        return nodeBuffer[upto];
      }
      return NO_MORE_DOCS;
    }
  }
}
