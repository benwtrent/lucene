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
package org.apache.lucene.benchmark.jmh;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Benchmark comparing dot product operations used by different ScalarEncoding types in
 * Lucene104ScalarQuantizedVectorsFormat.
 *
 * <p>This benchmark measures the raw dot product performance for:
 *
 * <ul>
 *   <li>UNSIGNED_BYTE: 8-bit unsigned query × 8-bit unsigned doc
 *   <li>SEVEN_BIT: 7-bit signed query × 7-bit signed doc
 *   <li>PACKED_NIBBLE: 4-bit query × 4-bit packed doc
 *   <li>SINGLE_BIT_QUERY_NIBBLE: 4-bit transposed query × 1-bit packed doc
 *   <li>DIBIT_QUERY_NIBBLE: 4-bit transposed query × 2-bit transposed doc
 *   <li>DIBIT_QUERY_BYTE: 8-bit transposed query × 2-bit transposed doc
 * </ul>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 4, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 3,
    jvmArgsAppend = {"-Xmx2g", "-Xms2g", "-XX:+AlwaysPreTouch"})
public class ScalarQuantizationDotProductBenchmark {

  @Param({"1024"})
  int size;

  // UNSIGNED_BYTE / SEVEN_BIT: raw byte arrays
  private byte[] bytesA;
  private byte[] bytesB;

  // PACKED_NIBBLE: 4-bit query unpacked, 4-bit doc packed
  private byte[] nibbleQuery;
  private byte[] nibbleDocPacked;

  // SINGLE_BIT_QUERY_NIBBLE: 4-bit query transposed (4 stripes), 1-bit doc packed
  private byte[] nibbleQueryTransposed;
  private byte[] binaryDocPacked;

  // DIBIT_QUERY_NIBBLE / DIBIT_QUERY_BYTE: 2-bit doc transposed (2 stripes)
  private byte[] dibitDocTransposed;

  // DIBIT_QUERY_BYTE: 8-bit query transposed (8 stripes)
  private byte[] byteQueryTransposed;

  @Setup(Level.Iteration)
  public void init() {
    ThreadLocalRandom random = ThreadLocalRandom.current();

    // --- UNSIGNED_BYTE / SEVEN_BIT ---
    bytesA = new byte[size];
    bytesB = new byte[size];
    random.nextBytes(bytesA);
    random.nextBytes(bytesB);

    // --- PACKED_NIBBLE ---
    nibbleQuery = new byte[size];
    byte[] nibbleDoc = new byte[size];
    for (int i = 0; i < size; i++) {
      nibbleQuery[i] = (byte) random.nextInt(16);
      nibbleDoc[i] = (byte) random.nextInt(16);
    }
    // Pack the doc nibbles
    nibbleDocPacked = new byte[(size + 1) >> 1];
    for (int i = 0; i < nibbleDocPacked.length; i++) {
      int high = nibbleDoc[i];
      int low = (i + nibbleDocPacked.length < size) ? nibbleDoc[i + nibbleDocPacked.length] : 0;
      nibbleDocPacked[i] = (byte) ((high << 4) | low);
    }

    // --- SINGLE_BIT_QUERY_NIBBLE ---
    // Transpose 4-bit query into 4 stripes
    int packedSize = (size + 7) / 8;
    nibbleQueryTransposed = new byte[packedSize * 4];
    OptimizedScalarQuantizer.transposeHalfByte(nibbleQuery, nibbleQueryTransposed);

    // Pack binary doc (1-bit values)
    byte[] binaryDoc = new byte[size];
    for (int i = 0; i < size; i++) {
      binaryDoc[i] = (byte) random.nextInt(2);
    }
    binaryDocPacked = new byte[packedSize];
    OptimizedScalarQuantizer.packAsBinary(binaryDoc, binaryDocPacked);

    // --- DIBIT_QUERY_NIBBLE / DIBIT_QUERY_BYTE ---
    // Transpose 2-bit doc into 2 stripes
    byte[] dibitDoc = new byte[size];
    for (int i = 0; i < size; i++) {
      dibitDoc[i] = (byte) random.nextInt(4);
    }
    dibitDocTransposed = new byte[packedSize * 2];
    OptimizedScalarQuantizer.transposeDibit(dibitDoc, dibitDocTransposed);

    // --- DIBIT_QUERY_BYTE ---
    // Transpose 8-bit query into 8 stripes
    byte[] byteQuery = new byte[size];
    for (int i = 0; i < size; i++) {
      byteQuery[i] = (byte) random.nextInt(256);
    }
    byteQueryTransposed = new byte[packedSize * 8];
    OptimizedScalarQuantizer.transposeByte(byteQuery, byteQueryTransposed);
  }

  // ==================== UNSIGNED_BYTE ====================

  @Benchmark
  public int uint8DotProductScalar() {
    return VectorUtil.uint8DotProduct(bytesA, bytesB);
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public int uint8DotProductVector() {
    return VectorUtil.uint8DotProduct(bytesA, bytesB);
  }

  // ==================== SEVEN_BIT ====================

  @Benchmark
  public int int7DotProductScalar() {
    return VectorUtil.dotProduct(bytesA, bytesB);
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public int int7DotProductVector() {
    return VectorUtil.dotProduct(bytesA, bytesB);
  }

  // ==================== PACKED_NIBBLE ====================

  @Benchmark
  public int int4DotProductPackedScalar() {
    return VectorUtil.int4DotProductSinglePacked(nibbleQuery, nibbleDocPacked);
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public int int4DotProductPackedVector() {
    return VectorUtil.int4DotProductSinglePacked(nibbleQuery, nibbleDocPacked);
  }

  // ==================== SINGLE_BIT_QUERY_NIBBLE ====================

  @Benchmark
  public long int4BitDotProductScalar() {
    return VectorUtil.int4BitDotProduct(nibbleQueryTransposed, binaryDocPacked);
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public long int4BitDotProductVector() {
    return VectorUtil.int4BitDotProduct(nibbleQueryTransposed, binaryDocPacked);
  }

  // ==================== DIBIT_QUERY_NIBBLE ====================

  @Benchmark
  public long int4DibitDotProductScalar() {
    return VectorUtil.int4DibitDotProduct(nibbleQueryTransposed, dibitDocTransposed);
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public long int4DibitDotProductVector() {
    return VectorUtil.int4DibitDotProduct(nibbleQueryTransposed, dibitDocTransposed);
  }

  // ==================== DIBIT_QUERY_BYTE ====================

  @Benchmark
  public long int8DibitDotProductScalar() {
    return VectorUtil.int8DibitDotProduct(byteQueryTransposed, dibitDocTransposed);
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public long int8DibitDotProductVector() {
    return VectorUtil.int8DibitDotProduct(byteQueryTransposed, dibitDocTransposed);
  }
}
