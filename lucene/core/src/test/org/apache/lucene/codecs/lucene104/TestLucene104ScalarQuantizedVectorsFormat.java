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
package org.apache.lucene.codecs.lucene104;

import static java.lang.String.format;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.oneOf;

import java.io.IOException;
import java.util.Locale;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.Before;

public class TestLucene104ScalarQuantizedVectorsFormat extends BaseKnnVectorsFormatTestCase {

  private ScalarEncoding encoding;
  private KnnVectorsFormat format;
  private float absTolerance;

  @Before
  @Override
  public void setUp() throws Exception {
    var encodingValues = ScalarEncoding.values();
    encoding = encodingValues[random().nextInt(encodingValues.length)];
    absTolerance =
        switch (encoding) {
          case UNSIGNED_BYTE, SEVEN_BIT -> 0.001f;
          case PACKED_NIBBLE -> 0.003f;
          case SINGLE_BIT_QUERY_NIBBLE -> 0.015f;
        };
    format = new Lucene104ScalarQuantizedVectorsFormat(encoding);
    super.setUp();
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(format);
  }

  public void testMergeScoreConsistency() throws IOException {
    float[] queryVector = new float[] {-0.5f, 90f, -10f, 14.8f};
    float[][] vectors =
        new float[][] {
          new float[] {230f, 300.33f, -34.8988f, 15.555f},
          new float[] {-0.5f, 100f, -13f, 14.8f},
          queryVector.clone()
        };

    for (var similarityFunction : VectorSimilarityFunction.values()) {
      if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
        // normalize vectors for dot product
        float[][] normalizedVectors = new float[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
          normalizedVectors[i] = vectors[i].clone();
          VectorUtil.l2normalize(normalizedVectors[i]);
        }
        float[] normalizedQueryVector = queryVector.clone();
        VectorUtil.l2normalize(normalizedQueryVector);
        assertMergeScoreConsistency(normalizedVectors, normalizedQueryVector, similarityFunction);
      } else {
        assertMergeScoreConsistency(vectors, queryVector, similarityFunction);
      }
    }
  }

  private void assertMergeScoreConsistency(
      float[][] vectors, float[] queryVector, VectorSimilarityFunction similarityFunction)
      throws IOException {
    KnnFloatVectorField knnField = new KnnFloatVectorField("vec", vectors[0], similarityFunction);
    StringField idField = new StringField("id", "0", Field.Store.YES);
    try (Directory dir = newDirectory()) {
      String[] ids = new String[3];
      float[] scores = new float[3];
      try (IndexWriter w =
          new IndexWriter(dir, newIndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
        int id = 0;
        for (float[] v : vectors) {
          Document doc = new Document();
          knnField.setVectorValue(v);
          idField.setStringValue(Integer.toString(id++));
          doc.add(knnField);
          doc.add(idField);
          w.addDocument(doc);
          w.flush();
        }
        try (IndexReader reader = DirectoryReader.open(w)) {
          IndexSearcher searcher = new IndexSearcher(reader);
          StoredFields fields = reader.storedFields();
          TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", queryVector, 3), 3);
          for (int i = 0; i < td.scoreDocs.length; i++) {
            scores[i] = td.scoreDocs[i].score;
            ids[i] = fields.document(td.scoreDocs[i].doc).getField("id").stringValue();
          }
        }
      }
      try (IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
        w.forceMerge(1);
        try (IndexReader reader = DirectoryReader.open(w)) {
          assertEquals(1, reader.leaves().size());
          StoredFields fields = reader.storedFields();
          IndexSearcher searcher = new IndexSearcher(reader);
          TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", queryVector, 3), 3);
          for (int i = 0; i < td.scoreDocs.length; i++) {
            String docId = fields.document(td.scoreDocs[i].doc).getField("id").stringValue();
            assertEquals(
                "encoding: "
                    + encoding
                    + " space: "
                    + similarityFunction
                    + " expected: ["
                    + ids[i]
                    + "]["
                    + scores[i]
                    + "] actual ["
                    + docId
                    + "]["
                    + td.scoreDocs[i].score
                    + "]",
                ids[i],
                docId);
            float tolerance =
                Math.max(absTolerance, absTolerance * Math.max(scores[i], td.scoreDocs[i].score));
            float absDiff = Math.abs(scores[i] - td.scoreDocs[i].score);
            assertTrue(
                "encoding: "
                    + encoding
                    + " space: "
                    + similarityFunction
                    + " expected score: "
                    + scores[i]
                    + " within "
                    + tolerance
                    + " but got: "
                    + td.scoreDocs[i].score
                    + " absDiff: "
                    + absDiff,
                absDiff <= tolerance);
          }
        }
      }
    }
  }

  public void testSearch() throws Exception {
    String fieldName = "field";
    int numVectors = random().nextInt(99, 500);
    int dims = random().nextInt(4, 65);
    float[] vector = randomVector(dims);
    VectorSimilarityFunction similarityFunction = randomSimilarity();
    KnnFloatVectorField knnField = new KnnFloatVectorField(fieldName, vector, similarityFunction);
    IndexWriterConfig iwc = newIndexWriterConfig();
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < numVectors; i++) {
          Document doc = new Document();
          knnField.setVectorValue(randomVector(dims));
          doc.add(knnField);
          w.addDocument(doc);
        }
        w.commit();

        try (IndexReader reader = DirectoryReader.open(w)) {
          IndexSearcher searcher = new IndexSearcher(reader);
          final int k = random().nextInt(5, 50);
          float[] queryVector = randomVector(dims);
          Query q = new KnnFloatVectorQuery(fieldName, queryVector, k);
          TopDocs collectedDocs = searcher.search(q, k);
          assertEquals(k, collectedDocs.totalHits.value());
          assertEquals(TotalHits.Relation.EQUAL_TO, collectedDocs.totalHits.relation());
        }
      }
    }
  }

  public void testToString() {
    FilterCodec customCodec =
        new FilterCodec("foo", Codec.getDefault()) {
          @Override
          public KnnVectorsFormat knnVectorsFormat() {
            return new Lucene104ScalarQuantizedVectorsFormat();
          }
        };
    String expectedPattern =
        "Lucene104ScalarQuantizedVectorsFormat("
            + "name=Lucene104ScalarQuantizedVectorsFormat, "
            + "encoding=UNSIGNED_BYTE, "
            + "flatVectorScorer=Lucene104ScalarQuantizedVectorScorer(nonQuantizedDelegate=%s()), "
            + "rawVectorFormat=Lucene99FlatVectorsFormat(vectorsScorer=%s()))";
    var defaultScorer =
        format(Locale.ROOT, expectedPattern, "DefaultFlatVectorScorer", "DefaultFlatVectorScorer");
    var memSegScorer =
        format(
            Locale.ROOT,
            expectedPattern,
            "Lucene99MemorySegmentFlatVectorsScorer",
            "Lucene99MemorySegmentFlatVectorsScorer");
    assertThat(customCodec.knnVectorsFormat().toString(), is(oneOf(defaultScorer, memSegScorer)));
  }

  @Override
  public void testRandomWithUpdatesAndGraph() {
    // graph not supported
  }

  @Override
  public void testSearchWithVisitedLimit() {
    // visited limit is not respected, as it is brute force search
  }

  public void testQuantizedVectorsWriteAndRead() throws IOException {
    String fieldName = "field";
    int numVectors = random().nextInt(99, 500);
    int dims = random().nextInt(4, 65);

    float[] vector = randomVector(dims);
    VectorSimilarityFunction similarityFunction = randomSimilarity();
    KnnFloatVectorField knnField = new KnnFloatVectorField(fieldName, vector, similarityFunction);
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
        for (int i = 0; i < numVectors; i++) {
          Document doc = new Document();
          knnField.setVectorValue(randomVector(dims));
          doc.add(knnField);
          w.addDocument(doc);
          if (i % 101 == 0) {
            w.commit();
          }
        }
        w.commit();
        w.forceMerge(1);

        try (IndexReader reader = DirectoryReader.open(w)) {
          LeafReader r = getOnlyLeafReader(reader);
          FloatVectorValues vectorValues = r.getFloatVectorValues(fieldName);
          assertEquals(vectorValues.size(), numVectors);
          QuantizedByteVectorValues qvectorValues =
              ((Lucene104ScalarQuantizedVectorsReader.ScalarQuantizedVectorValues) vectorValues)
                  .getQuantizedVectorValues();
          float[] centroid = qvectorValues.getCentroid();
          assertEquals(centroid.length, dims);

          OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarityFunction);
          byte[] scratch = new byte[encoding.getDiscreteDimensions(dims)];
          byte[] expectedVector = new byte[encoding.getDocPackedLength(scratch.length)];
          if (similarityFunction == VectorSimilarityFunction.COSINE) {
            vectorValues =
                new Lucene104ScalarQuantizedVectorsWriter.NormalizedFloatVectorValues(vectorValues);
          }
          KnnVectorValues.DocIndexIterator docIndexIterator = vectorValues.iterator();

          while (docIndexIterator.nextDoc() != NO_MORE_DOCS) {
            OptimizedScalarQuantizer.QuantizationResult corrections =
                quantizer.scalarQuantize(
                    vectorValues.vectorValue(docIndexIterator.index()),
                    scratch,
                    encoding.getBits(),
                    centroid);
            switch (encoding) {
              case UNSIGNED_BYTE, SEVEN_BIT ->
                  System.arraycopy(scratch, 0, expectedVector, 0, dims);
              case PACKED_NIBBLE ->
                  OffHeapScalarQuantizedVectorValues.packNibbles(scratch, expectedVector);
              case SINGLE_BIT_QUERY_NIBBLE ->
                  OptimizedScalarQuantizer.packAsBinary(scratch, expectedVector);
            }
            assertArrayEquals(expectedVector, qvectorValues.vectorValue(docIndexIterator.index()));
            var actualCorrections = qvectorValues.getCorrectiveTerms(docIndexIterator.index());
            assertEquals(corrections.lowerInterval(), actualCorrections.lowerInterval(), 0.00001f);
            assertEquals(corrections.upperInterval(), actualCorrections.upperInterval(), 0.00001f);
            assertEquals(
                corrections.additionalCorrection(),
                actualCorrections.additionalCorrection(),
                0.00001f);
            assertEquals(
                corrections.quantizedComponentSum(), actualCorrections.quantizedComponentSum());
          }
        }
      }
    }
  }
}
