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
package org.apache.lucene.search;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.SeededKnnCollector;
import org.apache.lucene.util.IOFunction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** A {@link KnnCollectorManager} that collects results with a timeout. */
public class SeededKnnCollectorManager implements KnnCollectorManager {
  private final KnnCollectorManager delegate;
  private final Weight seedWeight;
  private final int k;
  private final IOFunction<LeafReader, KnnVectorValues> vectorValuesSupplier;
  public SeededKnnCollectorManager(
    KnnCollectorManager delegate,
    String field,
    Query seededQuery,
    Query knnFilter,
    IndexSearcher indexSearcher,
    int k,
    IOFunction<LeafReader, KnnVectorValues> vectorValuesSupplier
  ) throws IOException {
    this.delegate = delegate;
    final Weight seedWeight;
    if (seededQuery != null) {
      BooleanQuery.Builder booleanSeedQueryBuilder =
        new BooleanQuery.Builder()
          .add(seededQuery, BooleanClause.Occur.MUST)
          .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER);
      if (knnFilter != null) {
        booleanSeedQueryBuilder.add(knnFilter, BooleanClause.Occur.FILTER);
      }
      Query seedRewritten = indexSearcher.rewrite(booleanSeedQueryBuilder.build());
      seedWeight = indexSearcher.createWeight(seedRewritten, ScoreMode.TOP_SCORES, 1f);
    } else {
      seedWeight = null;
    }
    this.seedWeight = seedWeight;
    this.k = k;
    this.vectorValuesSupplier = vectorValuesSupplier;
  }

  @Override
  public KnnCollector newCollector(int visitedLimit, LeafReaderContext ctx) throws IOException {
    if (seedWeight != null) {
      if (seedWeight == null) return null;
      // Execute the seed query
      TopScoreDocCollector seedCollector =
        new TopScoreDocCollectorManager(
          k /* numHits */,
          null /* after */,
          Integer.MAX_VALUE /* totalHitsThreshold */,
          false /* supportsConcurrency */)
          .newCollector();
      final LeafReader leafReader = ctx.reader();
      final LeafCollector leafCollector = seedCollector.getLeafCollector(ctx);
      if (leafCollector != null) {
        try {
          BulkScorer scorer = seedWeight.bulkScorer(ctx);
          if (scorer != null) {
            scorer.score(
              leafCollector,
              leafReader.getLiveDocs(),
              0 /* min */,
              DocIdSetIterator.NO_MORE_DOCS /* max */);
          }
          leafCollector.finish();
        } catch (
          @SuppressWarnings("unused")
          CollectionTerminatedException e) {
        }
      }

      TopDocs seedTopDocs = seedCollector.topDocs();
      if (seedTopDocs.totalHits.value() == 0) {
        return delegate.newCollector(visitedLimit, ctx);
      }
      KnnVectorValues vectorValues = vectorValuesSupplier.apply(leafReader);
      KnnVectorValues.DocIndexIterator indexIterator = vectorValues.iterator();
      DocIdSetIterator seedDocs = new MappedDISI(indexIterator, new TopDocsDISI(seedTopDocs));
      return new SeededKnnCollector(delegate.newCollector(visitedLimit, ctx), seedDocs);
    } else {
      return delegate.newCollector(visitedLimit, ctx);
    }
  }

  public static class MappedDISI extends DocIdSetIterator {
    KnnVectorValues.DocIndexIterator indexedDISI;
    DocIdSetIterator sourceDISI;

    public MappedDISI(KnnVectorValues.DocIndexIterator indexedDISI, DocIdSetIterator sourceDISI) {
      this.indexedDISI = indexedDISI;
      this.sourceDISI = sourceDISI;
    }

    /**
     * Advances the source iterator to the first document number that is greater than or equal to
     * the provided target and returns the corresponding index.
     */
    @Override
    public int advance(int target) throws IOException {
      int newTarget = sourceDISI.advance(target);
      if (newTarget != NO_MORE_DOCS) {
        indexedDISI.advance(newTarget);
      }
      return docID();
    }

    @Override
    public long cost() {
      return this.sourceDISI.cost();
    }

    @Override
    public int docID() {
      if (indexedDISI.docID() == NO_MORE_DOCS || sourceDISI.docID() == NO_MORE_DOCS) {
        return NO_MORE_DOCS;
      }
      return indexedDISI.index();
    }

    /** Advances to the next document in the source iterator and returns the corresponding index. */
    @Override
    public int nextDoc() throws IOException {
      int newTarget = sourceDISI.nextDoc();
      if (newTarget != NO_MORE_DOCS) {
        indexedDISI.advance(newTarget);
      }
      return docID();
    }
  }

  private static class TopDocsDISI extends DocIdSetIterator {
    private final List<Integer> sortedDocIdList;
    private int idx = -1;

    public TopDocsDISI(TopDocs topDocs) {
      sortedDocIdList = new ArrayList<Integer>(topDocs.scoreDocs.length);
      for (int i = 0; i < topDocs.scoreDocs.length; i++) {
        sortedDocIdList.add(topDocs.scoreDocs[i].doc);
      }
      Collections.sort(sortedDocIdList);
    }

    @Override
    public int advance(int target) throws IOException {
      return slowAdvance(target);
    }

    @Override
    public long cost() {
      return sortedDocIdList.size();
    }

    @Override
    public int docID() {
      if (idx == -1) {
        return -1;
      } else if (idx >= sortedDocIdList.size()) {
        return DocIdSetIterator.NO_MORE_DOCS;
      } else {
        return sortedDocIdList.get(idx);
      }
    }

    @Override
    public int nextDoc() throws IOException {
      idx += 1;
      return docID();
    }
  }
}
