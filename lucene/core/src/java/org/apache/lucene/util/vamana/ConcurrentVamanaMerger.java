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

import java.io.IOException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/**
 * This merger merges graph in a concurrent manner, by using {@link VamanaConcurrentMergeBuilder}
 */
public class ConcurrentVamanaMerger extends IncrementalVamanaGraphMerger {

  private final TaskExecutor exec;
  private final int numWorker;

  /**
   * @param fieldInfo FieldInfo for the field being merged
   */
  public ConcurrentVamanaMerger(
      FieldInfo fieldInfo,
      RandomVectorScorerSupplier scorerSupplier,
      int M,
      int beamWidth,
      float alpha,
      TaskExecutor exec,
      int numWorker) {
    super(fieldInfo, scorerSupplier, M, beamWidth, alpha);
    this.exec = exec;
    this.numWorker = numWorker;
  }

  @Override
  protected VamanaBuilder createBuilder(DocIdSetIterator mergedVectorIterator, int maxOrd)
      throws IOException {
    return new VamanaConcurrentMergeBuilder(
        exec,
        numWorker,
        scorerSupplier,
        M,
        beamWidth,
        alpha,
        new OnHeapVamanaGraph(M, maxOrd),
        null);
  }
}
