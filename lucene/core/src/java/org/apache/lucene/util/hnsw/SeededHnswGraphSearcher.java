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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.MappedDISI;
import org.apache.lucene.util.Bits;

/**
 * A {@link HnswGraphSearcher} that uses a set of seed ordinals to initiate the search.
 *
 * @lucene.experimental
 */
final class SeededHnswGraphSearcher extends AbstractHnswGraphSearcher {

  private final AbstractHnswGraphSearcher delegate;
  private final int[] seedOrds;
  private final float[] seedScores;

  static SeededHnswGraphSearcher fromEntryPoints(
      AbstractHnswGraphSearcher delegate, int numEps, DocIdSetIterator eps, int graphSize)
      throws IOException {
    if (numEps <= 0) {
      throw new IllegalArgumentException("The number of entry points must be > 0");
    }
    float[] seedScores = null;
    MappedDISI.Scored scored = null;
    if (eps instanceof MappedDISI.Scored scoredInstance) {
      seedScores = new float[numEps];
      scored = scoredInstance;
    }
    int[] entryPoints = new int[numEps];
    int idx = 0;
    while (idx < entryPoints.length) {
      int entryPointOrdInt = eps.nextDoc();
      if (entryPointOrdInt == NO_MORE_DOCS) {
        throw new IllegalArgumentException(
            "The number of entry points provided is less than the number of entry points requested");
      }
      assert entryPointOrdInt < graphSize;
      if (scored != null) {
        seedScores[idx] = scored.score();
      }
      entryPoints[idx++] = entryPointOrdInt;
    }
    return new SeededHnswGraphSearcher(delegate, entryPoints, seedScores);
  }

  SeededHnswGraphSearcher(AbstractHnswGraphSearcher delegate, int[] seedOrds, float[] seedScores) {
    this.delegate = delegate;
    this.seedOrds = seedOrds;
    this.seedScores = seedScores;
  }

  @Override
  void searchLevel(
      KnnCollector results,
      RandomVectorScorer scorer,
      int level,
      int[] eps,
      float[] epsScores,
      int epCount,
      HnswGraph graph,
      Bits acceptOrds)
      throws IOException {
    delegate.searchLevel(results, scorer, level, eps, epsScores, epCount, graph, acceptOrds);
  }

  @Override
  EntryPoints findBestEntryPoint(
      RandomVectorScorer scorer, HnswGraph graph, KnnCollector collector) {
    return new EntryPoints(seedOrds, seedScores);
  }
}
