package org.apache.lucene.search.knn;

import org.apache.lucene.search.DocIdSetIterator;

public interface EntryPointProvider {
  DocIdSetIterator entryPoints();
}
