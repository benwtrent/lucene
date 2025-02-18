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

import java.io.IOException;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;

/**
 * AbstractHnswGraphSearcher is the base class for HnswGraphSearcher implementations.
 *
 * @lucene.experimental
 */
abstract class AbstractHnswGraphSearcher {

  /**
   * Graph entry points, with optional scores
   *
   * @param ordinals the vector ordinals for the entry point
   * @param scores the scores
   */
  record EntryPoints(int[] ordinals, float[] scores) {}

  static final EntryPoints UNK_EP = new EntryPoints(new int[] {}, new float[] {});

  /**
   * Search a given level of the graph starting at the given entry points.
   *
   * @param results the collector to collect the results
   * @param scorer the scorer to compare the query with the nodes
   * @param level the level of the graph to search
   * @param eps the entry points to start the search from
   * @param scores the scores of the individual entry points, can be null
   * @param epCount the entry point count
   * @param graph the HNSWGraph
   * @param acceptOrds the ordinals to accept for the results
   */
  abstract void searchLevel(
      KnnCollector results,
      RandomVectorScorer scorer,
      int level,
      final int[] eps,
      final float[] scores,
      int epCount,
      HnswGraph graph,
      Bits acceptOrds)
      throws IOException;

  /**
   * Search a given level of the graph starting at the given entry points.
   *
   * @param results the collector to collect the results
   * @param scorer the scorer to compare the query with the nodes
   * @param level the level of the graph to search
   * @param eps the entry points to start the search from
   * @param scores the scores of the individual entry points, can be null
   * @param graph the HNSWGraph
   * @param acceptOrds the ordinals to accept for the results
   */
  final void searchLevel(
      KnnCollector results,
      RandomVectorScorer scorer,
      int level,
      final int[] eps,
      final float[] scores,
      HnswGraph graph,
      Bits acceptOrds)
      throws IOException {
    searchLevel(results, scorer, level, eps, scores, eps.length, graph, acceptOrds);
  }

  /**
   * Function to find the best entry point from which to search the zeroth graph layer.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param graph the HNSWGraph
   * @param collector the knn result collector
   * @return the best entry point, `-1` indicates graph entry node not set, or visitation limit
   *     exceeded
   * @throws IOException When accessing the vectors or graph fails
   */
  abstract EntryPoints findBestEntryPoint(
      RandomVectorScorer scorer, HnswGraph graph, KnnCollector collector) throws IOException;

  /**
   * Search the graph for the given scorer. Gathering results in the provided collector that pass
   * the provided acceptOrds.
   *
   * @param results the collector to collect the results
   * @param scorer the scorer to compare the query with the nodes
   * @param graph the HNSWGraph
   * @param acceptOrds the ordinals to accept for the results
   * @throws IOException When accessing the vectors or graph fails
   */
  public void search(
      KnnCollector results, RandomVectorScorer scorer, HnswGraph graph, Bits acceptOrds)
      throws IOException {
    EntryPoints eps = findBestEntryPoint(scorer, graph, results);
    assert eps != null;
    if (eps == UNK_EP) {
      return;
    }
    searchLevel(results, scorer, 0, eps.ordinals, eps.scores, graph, acceptOrds);
  }
}
