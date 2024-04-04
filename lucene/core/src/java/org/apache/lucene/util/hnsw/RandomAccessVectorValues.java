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

package org.apache.lucene.util.hnsw;

import java.io.IOException;
import java.util.List;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.Bits;

/**
 * Provides random access to vectors by dense ordinal. This interface is used by HNSW-based
 * implementations of KNN search.
 *
 * @lucene.experimental
 */
public interface RandomAccessVectorValues<T> {

  /** Return the number of vector values */
  int size();

  /** Return the dimension of the returned vector values */
  int dimension();

  /**
   * Return the vector value indexed at the given ordinal.
   *
   * @param targetOrd a valid ordinal, &ge; 0 and &lt; {@link #size()}.
   */
  T vectorValue(int targetOrd) throws IOException;

  /**
   * Creates a new copy of this {@link RandomAccessVectorValues}. This is helpful when you need to
   * access different values at once, to avoid overwriting the underlying float vector returned by
   * {@link RandomAccessVectorValues#vectorValue}.
   */
  RandomAccessVectorValues<T> copy() throws IOException;

  /**
   * Translates vector ordinal to the correct document ID. By default, this is an identity function.
   *
   * @param ord the vector ordinal
   * @return the document Id for that vector ordinal
   */
  default int ordToDoc(int ord) {
    return ord;
  }

  /**
   * Returns the {@link Bits} representing live documents. By default, this is an identity function.
   *
   * @param acceptDocs the accept docs
   * @return the accept docs
   */
  default Bits getAcceptOrds(Bits acceptDocs) {
    return acceptDocs;
  }

  /** Returns RandomAccessVectorValues that wraps a list of float[] vectors. */
  static RandomAccessVectorValues<float[]> fromFloatVectorList(
      List<float[]> vectors, int dimension) {
    return new RandomAccessVectorValues<>() {
      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int dimension() {
        return dimension;
      }

      @Override
      public float[] vectorValue(int targetOrd) {
        return vectors.get(targetOrd);
      }

      @Override
      public RandomAccessVectorValues<float[]> copy() {
        return this;
      }
    };
  }

  /** Returns RandomAccessVectorValues that wraps a list of byte[] vectors. */
  static RandomAccessVectorValues<byte[]> fromByteVectorList(List<byte[]> vectors, int dimension) {
    return new RandomAccessVectorValues<>() {
      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int dimension() {
        return dimension;
      }

      @Override
      public byte[] vectorValue(int targetOrd) {
        return vectors.get(targetOrd);
      }

      @Override
      public RandomAccessVectorValues<byte[]> copy() {
        return this;
      }
    };
  }

  /**
   * Returns RandomAccessVectorValues that wraps the provided byte vector values. The random access,
   * however, is only forward and depends on the iteration order of the provided {@link
   * ByteVectorValues}.
   */
  static RandomAccessVectorValues<byte[]> fromByteVectorValues(ByteVectorValues values) {
    return new RandomAccessVectorValues<>() {
      @Override
      public byte[] vectorValue(int docID) throws IOException {
        if (values.docID() != docID) {
          throw new IllegalArgumentException(
              "docID must be advanced in order by the caller on the ByteVectorValues");
        }
        return values.vectorValue();
      }

      @Override
      public RandomAccessVectorValues<byte[]> copy() throws IOException {
        throw new UnsupportedOperationException();
      }

      @Override
      public int dimension() {
        return values.dimension();
      }

      @Override
      public int size() {
        return values.size();
      }
    };
  }

  /**
   * Returns RandomAccessVectorValues that wraps the provided byte vector values. The random access,
   * however, is only forward and depends on the iteration order of the provided {@link
   * FloatVectorValues}.
   */
  static RandomAccessVectorValues<float[]> fromFloatVectorValues(FloatVectorValues values) {
    return new RandomAccessVectorValues<>() {
      @Override
      public float[] vectorValue(int docID) throws IOException {
        if (values.docID() != docID) {
          throw new IllegalArgumentException(
              "docID must be advanced in order by the caller on the FloatVectorValues");
        }
        return values.vectorValue();
      }

      @Override
      public RandomAccessVectorValues<float[]> copy() throws IOException {
        throw new UnsupportedOperationException();
      }

      @Override
      public int dimension() {
        return values.dimension();
      }

      @Override
      public int size() {
        return values.size();
      }
    };
  }
}
