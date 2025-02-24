/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.search.knn;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/** A {@link KnnFloatVectorQuery} that uses the IVF search strategy. */
public class IVFKnnFloatVectorQuery extends KnnFloatVectorQuery {

  private final int nprobe;

  public IVFKnnFloatVectorQuery(String field, float[] query, int k, Query filter, int nProbe) {
    super(field, query, k, filter, new IVFKnnSearchStrategy(nProbe));
    this.nprobe = nProbe;
  }

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    return super.rewrite(indexSearcher);
  }

  @Override
  protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
    long totalClustersVisisted = 0;
    for (TopDocs topDocs : perLeafResults) {
      if (topDocs instanceof IVFCollectorManager.ClusterCountedTopDocs cctd) {
        totalClustersVisisted += cctd.clusterCount;
      }
    }
    return super.mergeLeafResults(perLeafResults);
  }

  @Override
  protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
    return new IVFCollectorManager(k);
  }

  private class IVFCollectorManager implements KnnCollectorManager {
    private final int k;

    public IVFCollectorManager(int k) {
      this.k = k;
    }

    @Override
    public KnnCollector newCollector(
        int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
        throws IOException {
      return new ClusterCountingTopKnnCollector(
          new TopKnnCollector(k, visitedLimit, new IVFKnnSearchStrategy(nprobe)));
    }

    static class ClusterCountingTopKnnCollector extends KnnCollector.Decorator {
      public ClusterCountingTopKnnCollector(KnnCollector collector) {
        super(collector);
      }

      @Override
      public TopDocs topDocs() {
        return new ClusterCountedTopDocs(super.topDocs(), super.visitedClusterCount());
      }
    }

    static class ClusterCountedTopDocs extends TopDocs {
      private final long clusterCount;

      public ClusterCountedTopDocs(TopDocs topDocs, long clusterCount) {
        super(topDocs.totalHits, topDocs.scoreDocs);
        this.clusterCount = clusterCount;
      }
    }
  }
}
