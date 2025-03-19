/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.internal.vectorization;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.store.IndexInput;

final class MemorySegmentOSQVectorsScorer extends OSQVectorsScorer {

  private static final VectorSpecies<Integer> INT_SPECIES_128 = IntVector.SPECIES_128;
  private static final VectorSpecies<Long> LONG_SPECIES_128 = LongVector.SPECIES_128;
  private static final VectorSpecies<Long> LONG_SPECIES_256 = LongVector.SPECIES_256;

  private static final VectorSpecies<Byte> BYTE_SPECIES_128 = ByteVector.SPECIES_128;
  private static final VectorSpecies<Byte> BYTE_SPECIES_256 = ByteVector.SPECIES_256;

  private final MemorySegment memorySegment;

  MemorySegmentOSQVectorsScorer(IndexInput in, int length, MemorySegment memorySegment) {
    super(in, length);
    this.memorySegment = memorySegment;
  }

  @Override
  public long int4BitDotProduct(byte[] q) throws IOException {
    assert q.length == length * 4;
    // 128 / 8 == 16
    if (length >= 16 && PanamaVectorConstants.HAS_FAST_INTEGER_VECTORS) {
      if (PanamaVectorUtilSupport.VECTOR_BITSIZE >= 256) {
        return int4BitDotProduct256(q);
      } else if (PanamaVectorUtilSupport.VECTOR_BITSIZE == 128) {
        return int4BitDotProduct128(q);
      }
    }
    return super.int4BitDotProduct(q);
  }

  private long int4BitDotProduct256(byte[] q) throws IOException {
    long subRet0 = 0;
    long subRet1 = 0;
    long subRet2 = 0;
    long subRet3 = 0;
    int i = 0;
    long offset = in.getFilePointer();
    if (length >= ByteVector.SPECIES_256.vectorByteSize() * 2) {
      int limit = ByteVector.SPECIES_256.loopBound(length);
      var sum0 = LongVector.zero(LONG_SPECIES_256);
      var sum1 = LongVector.zero(LONG_SPECIES_256);
      var sum2 = LongVector.zero(LONG_SPECIES_256);
      var sum3 = LongVector.zero(LONG_SPECIES_256);
      for (;
          i < limit;
          i += ByteVector.SPECIES_256.length(), offset += LONG_SPECIES_256.vectorByteSize()) {
        var vq0 = ByteVector.fromArray(BYTE_SPECIES_256, q, i).reinterpretAsLongs();
        var vq1 = ByteVector.fromArray(BYTE_SPECIES_256, q, i + length).reinterpretAsLongs();
        var vq2 = ByteVector.fromArray(BYTE_SPECIES_256, q, i + length * 2).reinterpretAsLongs();
        var vq3 = ByteVector.fromArray(BYTE_SPECIES_256, q, i + length * 3).reinterpretAsLongs();
        var vd =
            LongVector.fromMemorySegment(
                LONG_SPECIES_256, memorySegment, offset, ByteOrder.LITTLE_ENDIAN);
        sum0 = sum0.add(vq0.and(vd).lanewise(VectorOperators.BIT_COUNT));
        sum1 = sum1.add(vq1.and(vd).lanewise(VectorOperators.BIT_COUNT));
        sum2 = sum2.add(vq2.and(vd).lanewise(VectorOperators.BIT_COUNT));
        sum3 = sum3.add(vq3.and(vd).lanewise(VectorOperators.BIT_COUNT));
      }
      subRet0 += sum0.reduceLanes(VectorOperators.ADD);
      subRet1 += sum1.reduceLanes(VectorOperators.ADD);
      subRet2 += sum2.reduceLanes(VectorOperators.ADD);
      subRet3 += sum3.reduceLanes(VectorOperators.ADD);
    }

    if (length - i >= ByteVector.SPECIES_128.vectorByteSize()) {
      var sum0 = LongVector.zero(LONG_SPECIES_128);
      var sum1 = LongVector.zero(LONG_SPECIES_128);
      var sum2 = LongVector.zero(LONG_SPECIES_128);
      var sum3 = LongVector.zero(LONG_SPECIES_128);
      int limit = ByteVector.SPECIES_128.loopBound(length);
      for (;
          i < limit;
          i += ByteVector.SPECIES_128.length(), offset += LONG_SPECIES_128.vectorByteSize()) {
        var vq0 = ByteVector.fromArray(BYTE_SPECIES_128, q, i).reinterpretAsLongs();
        var vq1 = ByteVector.fromArray(BYTE_SPECIES_128, q, i + length).reinterpretAsLongs();
        var vq2 = ByteVector.fromArray(BYTE_SPECIES_128, q, i + length * 2).reinterpretAsLongs();
        var vq3 = ByteVector.fromArray(BYTE_SPECIES_128, q, i + length * 3).reinterpretAsLongs();
        var vd =
            LongVector.fromMemorySegment(
                LONG_SPECIES_128, memorySegment, offset, ByteOrder.LITTLE_ENDIAN);
        sum0 = sum0.add(vq0.and(vd).lanewise(VectorOperators.BIT_COUNT));
        sum1 = sum1.add(vq1.and(vd).lanewise(VectorOperators.BIT_COUNT));
        sum2 = sum2.add(vq2.and(vd).lanewise(VectorOperators.BIT_COUNT));
        sum3 = sum3.add(vq3.and(vd).lanewise(VectorOperators.BIT_COUNT));
      }
      subRet0 += sum0.reduceLanes(VectorOperators.ADD);
      subRet1 += sum1.reduceLanes(VectorOperators.ADD);
      subRet2 += sum2.reduceLanes(VectorOperators.ADD);
      subRet3 += sum3.reduceLanes(VectorOperators.ADD);
    }
    // tail as bytes
    in.seek(offset);
    for (; i < length; i++) {
      int dValue = in.readByte() & 0xFF;
      subRet0 += Integer.bitCount((q[i] & dValue) & 0xFF);
      subRet1 += Integer.bitCount((q[i + length] & dValue) & 0xFF);
      subRet2 += Integer.bitCount((q[i + 2 * length] & dValue) & 0xFF);
      subRet3 += Integer.bitCount((q[i + 3 * length] & dValue) & 0xFF);
    }
    return subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
  }

  private long int4BitDotProduct128(byte[] q) throws IOException {
    long subRet0 = 0;
    long subRet1 = 0;
    long subRet2 = 0;
    long subRet3 = 0;
    int i = 0;
    long offset = in.getFilePointer();

    var sum0 = IntVector.zero(INT_SPECIES_128);
    var sum1 = IntVector.zero(INT_SPECIES_128);
    var sum2 = IntVector.zero(INT_SPECIES_128);
    var sum3 = IntVector.zero(INT_SPECIES_128);
    int limit = ByteVector.SPECIES_128.loopBound(length);
    for (;
        i < limit;
        i += ByteVector.SPECIES_128.length(), offset += INT_SPECIES_128.vectorByteSize()) {
      var vd =
          IntVector.fromMemorySegment(
              INT_SPECIES_128, memorySegment, offset, ByteOrder.LITTLE_ENDIAN);
      var vq0 = ByteVector.fromArray(BYTE_SPECIES_128, q, i).reinterpretAsInts();
      var vq1 = ByteVector.fromArray(BYTE_SPECIES_128, q, i + length).reinterpretAsInts();
      var vq2 = ByteVector.fromArray(BYTE_SPECIES_128, q, i + length * 2).reinterpretAsInts();
      var vq3 = ByteVector.fromArray(BYTE_SPECIES_128, q, i + length * 3).reinterpretAsInts();
      sum0 = sum0.add(vd.and(vq0).lanewise(VectorOperators.BIT_COUNT));
      sum1 = sum1.add(vd.and(vq1).lanewise(VectorOperators.BIT_COUNT));
      sum2 = sum2.add(vd.and(vq2).lanewise(VectorOperators.BIT_COUNT));
      sum3 = sum3.add(vd.and(vq3).lanewise(VectorOperators.BIT_COUNT));
    }
    subRet0 += sum0.reduceLanes(VectorOperators.ADD);
    subRet1 += sum1.reduceLanes(VectorOperators.ADD);
    subRet2 += sum2.reduceLanes(VectorOperators.ADD);
    subRet3 += sum3.reduceLanes(VectorOperators.ADD);
    // tail as bytes
    in.seek(offset);
    for (; i < length; i++) {
      int dValue = in.readByte() & 0xFF;
      subRet0 += Integer.bitCount((dValue & q[i]) & 0xFF);
      subRet1 += Integer.bitCount((dValue & q[i + length]) & 0xFF);
      subRet2 += Integer.bitCount((dValue & q[i + 2 * length]) & 0xFF);
      subRet3 += Integer.bitCount((dValue & q[i + 3 * length]) & 0xFF);
    }
    return subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
  }
}
