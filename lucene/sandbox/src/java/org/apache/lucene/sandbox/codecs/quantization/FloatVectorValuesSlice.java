/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import java.io.IOException;
import java.util.stream.IntStream;
import org.apache.lucene.index.FloatVectorValues;

public class FloatVectorValuesSlice extends FloatVectorValues {

  final FloatVectorValues allValues;
  final int[] slice;

  FloatVectorValuesSlice(FloatVectorValues allValues, int[] slice) {
    this.allValues = allValues;
    this.slice = slice;
  }

  FloatVectorValuesSlice(FloatVectorValues allValues) {
    this.allValues = allValues;
    this.slice = IntStream.range(0, allValues.size()).toArray();
  }

  @Override
  public float[] vectorValue(int ord) throws IOException {
    return this.allValues.vectorValue(this.slice[ord]);
  }

  @Override
  public int dimension() {
    return this.allValues.dimension();
  }

  @Override
  public int size() {
    return slice.length;
  }

  @Override
  public FloatVectorValues copy() throws IOException {
    return new FloatVectorValuesSlice(this.allValues.copy(), this.slice);
  }
}
