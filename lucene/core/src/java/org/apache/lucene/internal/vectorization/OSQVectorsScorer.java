/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.internal.vectorization;

import java.io.IOException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitUtil;

public class OSQVectorsScorer {

  /** The wrapper {@link IndexInput}. */
  protected final IndexInput in;

  protected final int length;

  /** Sole constructor, called by sub-classes. */
  protected OSQVectorsScorer(IndexInput in, int length) {
    this.in = in;
    this.length = length;
  }

  public long int4BitDotProduct(byte[] q) throws IOException {
    assert q.length == length * 4;
    final int size = length;
    long subRet0 = 0;
    long subRet1 = 0;
    long subRet2 = 0;
    long subRet3 = 0;
    int r = 0;
    for (final int upperBound = size & -Long.BYTES; r < upperBound; r += Long.BYTES) {
      final long value = in.readLong();
      subRet0 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r) & value);
      subRet1 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r + size) & value);
      subRet2 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r + 2 * size) & value);
      subRet3 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r + 3 * size) & value);
    }
    for (final int upperBound = size & -Integer.BYTES; r < upperBound; r += Integer.BYTES) {
      final int value = in.readInt();
      subRet0 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r) & value);
      subRet1 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r + size) & value);
      subRet2 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r + 2 * size) & value);
      subRet3 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r + 3 * size) & value);
    }
    for (; r < size; r++) {
      final byte value = in.readByte();
      subRet0 += Integer.bitCount((q[r] & value) & 0xFF);
      subRet1 += Integer.bitCount((q[r + size] & value) & 0xFF);
      subRet2 += Integer.bitCount((q[r + 2 * size] & value) & 0xFF);
      subRet3 += Integer.bitCount((q[r + 3 * size] & value) & 0xFF);
    }
    return subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
  }
}
