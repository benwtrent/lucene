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

package org.apache.lucene.facet;

import java.io.IOException;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;

public class TestFacetUtils extends LuceneTestCase {

  public void testBasic() throws IOException {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random(), dir);
    int maxDocs = 256;

    // Create up to maxDocs number of documents with (id, val) fields having
    // string and numeric representation of the same value
    for (int i = 0; i < maxDocs; i++) {
      Document doc = new Document();
      doc.add(new StringField("id", String.valueOf(i), Field.Store.NO));
      doc.add(new NumericDocValuesField("val", i));
      writer.addDocument(doc);
    }

    // Pick a random number of docs to delete.
    // Actual deleted docs might be less than this because
    // of duplicates generated by random()
    int numDocsToDelete = random().nextInt(maxDocs);
    FixedBitSet deletedDocs = new FixedBitSet(maxDocs);

    // Delete a random number of randomly picked documents
    // Record what we deleted to make sure we don't find them
    // during iteration
    int actualDocsDeleted = 0;
    for (int i = 0; i < numDocsToDelete; i++) {
      int deletedDocId = random().nextInt(maxDocs);
      if (deletedDocs.get(deletedDocId) == true) {
        // we already deleted and recorded this doc-id
        continue;
      }
      deletedDocs.set(deletedDocId);
      writer.deleteDocuments(new Term("id", deletedDocId + ""));
      actualDocsDeleted++;
    }

    IndexSearcher searcher = newSearcher(writer.getReader());
    FacetsCollector fc = searcher.search(new MatchAllDocsQuery(), new FacetsCollectorManager());

    int visitedDocs = 0;
    DocIdSetIterator disi;

    for (FacetsCollector.MatchingDocs m : fc.getMatchingDocs()) {
      NumericDocValues numericDV = DocValues.getNumeric(m.context().reader(), "val");
      Bits liveDocs = m.context().reader().getLiveDocs();
      // Only use the liveDocsDISI if liveDocs is not null
      disi = (liveDocs == null) ? numericDV : FacetUtils.liveDocsDISI(numericDV, liveDocs);

      // Iterating over disi should only give values for live docs
      while (disi.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        visitedDocs++;
        int val = (int) numericDV.longValue();
        assertTrue("Deleted doc " + val + " found during iteration", deletedDocs.get(val) == false);
      }
    }

    assertTrue(
        "Visited document ["
            + visitedDocs
            + "] != Live documents ["
            + (maxDocs - actualDocsDeleted)
            + "]",
        visitedDocs == maxDocs - actualDocsDeleted);
    writer.close();
    IOUtils.close(searcher.getIndexReader(), dir);
  }
}
