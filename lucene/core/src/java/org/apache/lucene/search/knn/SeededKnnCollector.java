package org.apache.lucene.search.knn;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;

public class SeededKnnCollector extends KnnCollector.Decorator implements EntryPointProvider {
  private final DocIdSetIterator entryPoints;

  public SeededKnnCollector(KnnCollector collector, DocIdSetIterator entryPoints) {
    super(collector);
    this.entryPoints = entryPoints;
  }

  @Override
  public DocIdSetIterator entryPoints() {
    return entryPoints;
  }
}
