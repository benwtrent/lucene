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
import java.nio.ByteBuffer;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

/** Read the quantized vector values. This supports both iterated and random access. */
public abstract class OffHeapBinaryQuantizedVectorValues extends QuantizedByteVectorValues
    implements RandomAccessVectorValues<byte[]> {

  protected final int dimension;
  protected final int size;
  protected final IndexInput slice;
  protected final byte[] binaryValue;
  protected final ByteBuffer byteBuffer;
  protected final int byteSize;
  protected int lastOrd = -1;

  OffHeapBinaryQuantizedVectorValues(int dimension, int size, IndexInput slice) {
    this.dimension = dimension;
    this.size = size;
    this.slice = slice;
    this.byteSize = dimension >> 3;
    byteBuffer = ByteBuffer.allocate(byteSize);
    binaryValue = byteBuffer.array();
  }

  @Override
  public float getScoreCorrectionConstant() throws IOException {
    return 0;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public byte[] vectorValue(int targetOrd) throws IOException {
    if (lastOrd == targetOrd) {
      return binaryValue;
    }
    slice.seek((long) targetOrd * byteSize);
    slice.readBytes(byteBuffer.array(), byteBuffer.arrayOffset(), byteSize);
    lastOrd = targetOrd;
    return binaryValue;
  }

  public static OffHeapBinaryQuantizedVectorValues load(
      OrdToDocDISIReaderConfiguration configuration,
      int dimension,
      int size,
      long quantizedVectorDataOffset,
      long quantizedVectorDataLength,
      IndexInput vectorData)
      throws IOException {
    if (configuration.isEmpty()) {
      return new EmptyOffHeapVectorValues(dimension);
    }
    IndexInput bytesSlice =
        vectorData.slice(
            "binary-quantized-vector-data", quantizedVectorDataOffset, quantizedVectorDataLength);
    if (configuration.isDense()) {
      return new DenseOffHeapVectorValues(dimension, size, bytesSlice);
    } else {
      return new SparseOffHeapVectorValues(configuration, dimension, size, vectorData, bytesSlice);
    }
  }

  /**
   * Dense vector values that are stored off-heap. This is the most common case when every doc has a
   * vector.
   */
  public static class DenseOffHeapVectorValues extends OffHeapBinaryQuantizedVectorValues {

    private int doc = -1;

    public DenseOffHeapVectorValues(int dimension, int size, IndexInput slice) {
      super(dimension, size, slice);
    }

    @Override
    public byte[] vectorValue() throws IOException {
      return vectorValue(doc);
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
      assert docID() < target;
      if (target >= size) {
        return doc = NO_MORE_DOCS;
      }
      return doc = target;
    }

    @Override
    public DenseOffHeapVectorValues copy() throws IOException {
      return new DenseOffHeapVectorValues(dimension, size, slice.clone());
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      return acceptDocs;
    }
  }

  private static class SparseOffHeapVectorValues extends OffHeapBinaryQuantizedVectorValues {
    private final DirectMonotonicReader ordToDoc;
    private final IndexedDISI disi;
    // dataIn was used to init a new IndexedDIS for #randomAccess()
    private final IndexInput dataIn;
    private final OrdToDocDISIReaderConfiguration configuration;

    public SparseOffHeapVectorValues(
        OrdToDocDISIReaderConfiguration configuration,
        int dimension,
        int size,
        IndexInput dataIn,
        IndexInput slice)
        throws IOException {
      super(dimension, size, slice);
      this.configuration = configuration;
      this.dataIn = dataIn;
      this.ordToDoc = configuration.getDirectMonotonicReader(dataIn);
      this.disi = configuration.getIndexedDISI(dataIn);
    }

    @Override
    public byte[] vectorValue() throws IOException {
      return vectorValue(disi.index());
    }

    @Override
    public int docID() {
      return disi.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      return disi.nextDoc();
    }

    @Override
    public int advance(int target) throws IOException {
      assert docID() < target;
      return disi.advance(target);
    }

    @Override
    public SparseOffHeapVectorValues copy() throws IOException {
      return new SparseOffHeapVectorValues(configuration, dimension, size, dataIn, slice.clone());
    }

    @Override
    public int ordToDoc(int ord) {
      return (int) ordToDoc.get(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      if (acceptDocs == null) {
        return null;
      }
      return new Bits() {
        @Override
        public boolean get(int index) {
          return acceptDocs.get(ordToDoc(index));
        }

        @Override
        public int length() {
          return size;
        }
      };
    }
  }

  private static class EmptyOffHeapVectorValues extends OffHeapBinaryQuantizedVectorValues {

    public EmptyOffHeapVectorValues(int dimension) {
      super(dimension, 0, null);
    }

    private int doc = -1;

    @Override
    public int dimension() {
      return super.dimension();
    }

    @Override
    public int size() {
      return 0;
    }

    @Override
    public byte[] vectorValue() {
      throw new UnsupportedOperationException();
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc + 1);
    }

    @Override
    public int advance(int target) {
      return doc = NO_MORE_DOCS;
    }

    @Override
    public EmptyOffHeapVectorValues copy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte[] vectorValue(int targetOrd) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int ordToDoc(int ord) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      return null;
    }
  }
}
