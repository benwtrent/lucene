package org.apache.lucene.search;


import org.apache.lucene.search.knn.KnnCollectorManager;

import java.io.IOException;

public class SeededKnnFloatVectorQuery extends KnnFloatVectorQuery {
  private final Query seed;
  public SeededKnnFloatVectorQuery(String field, float[] target, int k, Query filter, Query seed) {
    super(field, target, k, filter);
    this.seed = seed;
  }

  @Override
  protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) throws IOException {
    return new SeededKnnCollectorManager(super.getKnnCollectorManager(k, searcher), field, seed, filter, searcher, k, leaf -> leaf.getFloatVectorValues(field));
  }
}
