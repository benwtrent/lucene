package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

public class VectorsReaderWithOffset implements RandomAccessVectorValues.Floats {
  private final IndexInput slice;
  private final int size;
  private final int dim;
  private final int byteSize;
  private final int offset;
  private int lastOrd = -1;
  private final float[] value;

  public VectorsReaderWithOffset(IndexInput slice, int size, int dim, int offset) {
    this.slice = slice;
    this.size = size;
    this.dim = dim;
    // We assume that the start of ever vector entry includes an integer/float that indicates its
    // dimension count
    this.byteSize = (Float.BYTES * dim + Float.BYTES);
    this.offset = offset;
    value = new float[dim];
  }

  @Override
  public int dimension() {
    return dim;
  }

  @Override
  public IndexInput getSlice() {
    return slice;
  }

  @Override
  public int ordToDoc(int ord) {
    throw new IllegalStateException("Not supported");
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    throw new IllegalStateException("Not supported");
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int getVectorByteLength() {
    return byteSize;
  }

  @Override
  public float[] vectorValue(int targetOrd) throws IOException {
    if (lastOrd == targetOrd) {
      return value;
    }
    // Get to the appropriate vector for the ordinal, then skip the first 4 bytes storing its
    // dimension count
    long seekPos = (long) targetOrd * byteSize + offset;
    slice.seek(seekPos);
    slice.readFloats(value, 0, value.length);
    lastOrd = targetOrd;
    return value;
  }

  @Override
  public Floats copy() throws IOException {
    return new VectorsReaderWithOffset(slice, size, dim, offset);
  }
}
