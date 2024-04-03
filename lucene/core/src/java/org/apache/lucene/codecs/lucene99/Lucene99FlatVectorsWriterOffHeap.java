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

package org.apache.lucene.codecs.lucene99;

import java.io.IOException;
import org.apache.lucene.codecs.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/**
 * Writes vector values to index segments.
 *
 * @lucene.experimental
 */
public final class Lucene99FlatVectorsWriterOffHeap extends FlatVectorsWriter {

  private static final long SHALLLOW_RAM_BYTES_USED =
      RamUsageEstimator.shallowSizeOfInstance(Lucene99FlatVectorsWriterOffHeap.class);

  private final FlatVectorsWriter delegate;

  public Lucene99FlatVectorsWriterOffHeap(FlatVectorsWriter delegate) throws IOException {
    this.delegate = delegate;
  }

  @Override
  public FlatFieldVectorsWriter<?> addField(
      FieldInfo fieldInfo, KnnFieldVectorsWriter<?> indexWriter) throws IOException {
    return delegate.addField(fieldInfo, indexWriter);
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    delegate.flush(maxDoc, sortMap);
  }

  @Override
  public void finish() throws IOException {
    delegate.finish();
  }

  @Override
  public long ramBytesUsed() {
    return delegate.ramBytesUsed();
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    delegate.mergeOneField(fieldInfo, mergeState);
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    CloseableRandomVectorScorerSupplier supplier =
        delegate.mergeOneFieldToIndex(fieldInfo, mergeState);
    if (supplier.copy()
        instanceof RandomVectorScorerSupplier.FloatScoringSupplier floatScoringSupplier) {
      RandomAccessVectorValues<float[]> vectorValues = floatScoringSupplier.getVectors();
      if (vectorValues instanceof OffHeapFloatVectorValues offHeapFloatVectorValues) {
        // Do my off heap stuff, and be sure to close the original supplier
      }
    }
    return supplier;
  }

  @Override
  public void close() throws IOException {
    delegate.close();
  }
}
