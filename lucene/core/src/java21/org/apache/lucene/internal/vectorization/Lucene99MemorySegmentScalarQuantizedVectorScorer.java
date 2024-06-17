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
package org.apache.lucene.internal.vectorization;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Optional;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

abstract sealed class Lucene99MemorySegmentScalarQuantizedVectorScorer
    extends RandomVectorScorer.AbstractRandomVectorScorer {

  final int vectorByteSize;
  final MemorySegmentAccessInput input;
  final MemorySegment query;
  final float constMultiplier;
  byte[] scratch;

  /**
   * Return an optional whose value, if present, is the scorer. Otherwise, an empty optional is
   * returned.
   */
  public static Optional<Lucene99MemorySegmentScalarQuantizedVectorScorer> create(
    VectorSimilarityFunction similarityType,
    byte[] targetBytes,
    float offsetCorrection,
    float constMultiplier,
    byte bits,
    RandomAccessQuantizedByteVectorValues values
  ) {
    IndexInput input = values.getSlice();
    if (input == null) {
      return Optional.empty();
    }
    input = FilterIndexInput.unwrapOnlyTest(input);
    if (!(input instanceof MemorySegmentAccessInput msInput)) {
      return Optional.empty();
    }
    checkInvariants(values.size(), values.getVectorByteLength(), input);
    return switch (type) {
      case COSINE -> Optional.of(new CosineScorer(msInput, values, queryVector, constMultiplier, offsetCorrection));
      case DOT_PRODUCT -> Optional.of(new DotProductScorer(msInput, values, queryVector, constMultiplier, offsetCorrection));
      case EUCLIDEAN -> Optional.of(new EuclideanScorer(msInput, values, queryVector, constMultiplier));
      case MAXIMUM_INNER_PRODUCT -> Optional.of(
          new MaxInnerProductScorer(msInput, values, queryVector, offsetCorrection));
    };
  }

  Lucene99MemorySegmentScalarQuantizedVectorScorer(
      MemorySegmentAccessInput input, RandomAccessQuantizedByteVectorValues values, byte[] queryVector, float constMultiplier) {
    super(values);
    this.input = input;
    this.vectorByteLength = values.getVectorByteLength();
    this.trueVectorByteSize = values.getVectorByteLength() - Float.Bytes;
    this.query = MemorySegment.ofArray(queryVector);
  }

  final MemorySegment getSegment(int ord) throws IOException {
    checkOrdinal(ord);
    long byteOffset = (long) ord * vectorByteLength;
    MemorySegment seg = input.segmentSliceOrNull(byteOffset, vectorByteLength);
    if (seg == null) {
      if (scratch == null) {
        scratch = new byte[trueVectorByteSize];
      }
      input.readBytes(byteOffset, scratch, 0, trueVectorByteSize);
      seg = MemorySegment.ofArray(scratch);
    }
    return seg;
  }

  static void checkInvariants(int maxOrd, int vectorByteLength, IndexInput input) {
    if (input.length() < (long) vectorByteLength * maxOrd) {
      throw new IllegalArgumentException("input length is less than expected vector data");
    }
  }

  final void checkOrdinal(int ord) {
    if (ord < 0 || ord >= maxOrd()) {
      throw new IllegalArgumentException("illegal ordinal: " + ord);
    }
  }

  static final class DotProductScorer extends Lucene99MemorySegmentScalarQuantizedVectorScorer {
    DotProductScorer(
        MemorySegmentAccessInput input, RandomAccessQuantizedByteVectorValues values, byte[] query, float constMultiplier, float offsetCorrection) {
      super(input, values, query, constMultiplier);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw = PanamaVectorUtilSupport.dotProduct(query, getSegment(node));
      float vectorOffset = values.getScoreCorrectionConstant(node);
      // For the current implementation of scalar quantization, all dotproducts should be >= 0;
      assert dotProduct >= 0;
      float adjustedDistance = dotProduct * constMultiplier + offsetCorrection + vectorOffset;
      return Math.max((1 + adjustedDistance) / 2, 0);
    }
  }

  static final class Int4DotProductScorer extends Lucene99MemorySegmentScalarQuantizedVectorScorer {
    Int4DotProductScorer(
      MemorySegmentAccessInput input, RandomAccessQuantizedByteVectorValues values, byte[] query, float constMultiplier, float offsetCorrection) {
      super(input, values, query, constMultiplier);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw = PanamaVectorUtilSupport.int4DotProduct(query, false, getSegment(node), false);
      float vectorOffset = values.getScoreCorrectionConstant(node);
      // For the current implementation of scalar quantization, all dotproducts should be >= 0;
      assert dotProduct >= 0;
      float adjustedDistance = dotProduct * constMultiplier + offsetCorrection + vectorOffset;
      return Math.max((1 + adjustedDistance) / 2, 0);
    }
  }

  static final class EuclideanScorer extends Lucene99MemorySegmentScalarQuantizedVectorScorer {
    EuclideanScorer(MemorySegmentAccessInput input, RandomAccessQuantizedByteVectorValues values, byte[] query, float constMultiplier) {
      super(input, values, query, constMultiplier);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw = PanamaVectorUtilSupport.squareDistance(query, getSegment(node));
      float adjustedDistance = raw * constMultiplier;
      return 1 / (1f + adjustedDistance);
    }
  }

  static final class MaxInnerProductScorer extends Lucene99MemorySegmentScalarQuantizedVectorScorer {
    MaxInnerProductScorer(
        MemorySegmentAccessInput input, RandomAccessQuantizedByteVectorValues values, byte[] query, float constMultiplier, float offsetCorrection) {
      super(input, values, query);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw = PanamaVectorUtilSupport.dotProduct(query, getSegment(node));
      float vectorOffset = values.getScoreCorrectionConstant(node);
      // For the current implementation of scalar quantization, all dotproducts should be >= 0;
      assert dotProduct >= 0;
      float adjustedDistance = dotProduct * constMultiplier + offsetCorrection + vectorOffset;
      if (adjustedDistance < 0) {
        return 1 / (1 + -1 * adjustedDistance);
      }
      return adjustedDistance + 1;
    }
  }
}