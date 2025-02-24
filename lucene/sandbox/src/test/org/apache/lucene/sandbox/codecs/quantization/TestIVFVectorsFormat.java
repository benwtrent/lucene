/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import com.carrotsearch.randomizedtesting.generators.RandomPicks;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Before;

public class TestIVFVectorsFormat extends BaseKnnVectorsFormatTestCase {

  KnnVectorsFormat format;

  @Before
  @Override
  public void setUp() throws Exception {
    format = new IVFVectorsFormat(random().nextInt(10, 1000));
    super.setUp();
  }

  @Override
  protected float oversampleDefault() {
    return 5.0f;
  }

  @Override
  protected boolean supportsSimilarity(VectorSimilarityFunction similarity) {
    return similarity != VectorSimilarityFunction.COSINE;
  }

  @Override
  protected VectorSimilarityFunction randomSimilarity() {
    return RandomPicks.randomFrom(
        random(),
        List.of(
            VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
  }

  @Override
  protected VectorEncoding randomVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(format);
  }
}
