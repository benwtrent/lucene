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

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.ScalarQuantizer;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.vamana.VamanaGraph;
import org.apache.lucene.util.vamana.VamanaGraphSearcher;

/**
 * Reads vectors from the index segments along with index data structures supporting KNN search.
 *
 * @lucene.experimental
 */
public final class Lucene99VamanaVectorsReader extends KnnVectorsReader
    implements QuantizedVectorsReader {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(Lucene99VamanaVectorsFormat.class);

  private final Map<String, FieldEntry> fields = new HashMap<>();
  private final IndexInput vectorIndex;
  private final FlatVectorsReader flatVectorsReader;

  Lucene99VamanaVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader)
      throws IOException {
    this.flatVectorsReader = flatVectorsReader;
    boolean success = false;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene99VamanaVectorsFormat.META_EXTENSION);
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                Lucene99VamanaVectorsFormat.META_CODEC_NAME,
                Lucene99VamanaVectorsFormat.VERSION_START,
                Lucene99VamanaVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(meta, state.fieldInfos);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
      vectorIndex =
          openDataInput(
              state,
              versionMeta,
              Lucene99VamanaVectorsFormat.VECTOR_INDEX_EXTENSION,
              Lucene99VamanaVectorsFormat.VECTOR_INDEX_CODEC_NAME);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  private static IndexInput openDataInput(
      SegmentReadState state, int versionMeta, String fileExtension, String codecName)
      throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    IndexInput in = state.directory.openInput(fileName, state.context);
    boolean success = false;
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              codecName,
              Lucene99VamanaVectorsFormat.VERSION_START,
              Lucene99VamanaVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta="
                + versionMeta
                + ", "
                + codecName
                + "="
                + versionVectorData,
            in);
      }
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.name, fieldEntry);
    }
  }

  private void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dimension) {
      throw new IllegalStateException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + fieldEntry.dimension);
    }
  }

  private FieldEntry readField(IndexInput input) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(input);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    return new FieldEntry(input, vectorEncoding, similarityFunction);
  }

  @Override
  public long ramBytesUsed() {
    return Lucene99VamanaVectorsReader.SHALLOW_SIZE
        + RamUsageEstimator.sizeOfMap(
            fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class))
        + flatVectorsReader.ramBytesUsed();
  }

  @Override
  public void checkIntegrity() throws IOException {
    flatVectorsReader.checkIntegrity();
    CodecUtil.checksumEntireFile(vectorIndex);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return flatVectorsReader.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return flatVectorsReader.getByteVectorValues(field);
  }

  @Override
  public QuantizedByteVectorValues getQuantizedVectorValues(String fieldName) throws IOException {
    // TODO how can we iterate the quantized values in order?
    //  Can we do this from within the InGraph iterators?
    return null;
  }

  @Override
  public ScalarQuantizer getQuantizationState(String fieldName) {
    return fields.get(fieldName).scalarQuantizer;
  }

  VamanaGraph getGraph(FieldEntry fieldEntry) throws IOException {
    return new OffHeapVamanaGraph(fieldEntry, vectorIndex);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    FieldEntry fieldEntry = fields.get(field);

    if (fieldEntry.size() == 0
        || knnCollector.k() == 0
        || fieldEntry.vectorEncoding != VectorEncoding.FLOAT32) {
      return;
    }
    InGraphOffHeapQuantizedByteVectorValues vectorValues =
        InGraphOffHeapQuantizedByteVectorValues.load(fieldEntry, vectorIndex);
    RandomVectorScorer scorer =
        new ScalarQuantizedRandomVectorScorer(
            fieldEntry.similarityFunction, fieldEntry.scalarQuantizer, vectorValues, target);
    VamanaGraphSearcher.search(
        scorer,
        new OrdinalTranslatedKnnCollector(knnCollector, vectorValues::ordToDoc),
        getGraph(fieldEntry),
        // FIXME: support filtered
        //          vectorValues.getAcceptOrds(acceptDocs));
        acceptDocs,
        null);
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(flatVectorsReader, vectorIndex);
  }

  public static class FieldEntry implements Accountable {

    private static final long SHALLOW_SIZE =
        RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class);
    final VectorSimilarityFunction similarityFunction;
    final VectorEncoding vectorEncoding;
    final long vectorIndexOffset;
    final long vectorIndexLength;
    final int M;
    final int entryNode;
    final int dimension;
    final int size;
    final DirectMonotonicReader.Meta offsetsMeta;
    final long offsetsOffset;
    final int offsetsBlockShift;
    final long offsetsLength;
    final float lowerQuantile, upperQuantile;
    final ScalarQuantizer scalarQuantizer;

    final boolean isQuantized;

    FieldEntry(
        IndexInput meta, VectorEncoding vectorEncoding, VectorSimilarityFunction similarityFunction)
        throws IOException {
      this.similarityFunction = similarityFunction;
      this.vectorEncoding = vectorEncoding;
      this.isQuantized = meta.readByte() == 1;
      assert this.isQuantized;
      // Has int8 quantization
      lowerQuantile = Float.intBitsToFloat(meta.readInt());
      upperQuantile = Float.intBitsToFloat(meta.readInt());
      vectorIndexOffset = meta.readVLong();
      vectorIndexLength = meta.readVLong();
      dimension = meta.readVInt();
      scalarQuantizer =
          new ScalarQuantizer(
              lowerQuantile,
              upperQuantile,
              Lucene99ScalarQuantizedVectorsFormat.calculateDefaultConfidenceInterval(dimension));
      size = meta.readInt();

      // read node offsets
      M = meta.readVInt();
      if (size > 0) {
        entryNode = meta.readVInt();
        offsetsOffset = meta.readLong();
        offsetsBlockShift = meta.readVInt();
        offsetsMeta = DirectMonotonicReader.loadMeta(meta, size, offsetsBlockShift);
        offsetsLength = meta.readLong();
      } else {
        entryNode = -1;
        offsetsOffset = 0;
        offsetsBlockShift = 0;
        offsetsMeta = null;
        offsetsLength = 0;
      }
    }

    int size() {
      return size;
    }

    @Override
    public long ramBytesUsed() {
      return SHALLOW_SIZE + RamUsageEstimator.sizeOf(offsetsMeta);
    }
  }

  /** Read the nearest-neighbors graph from the index input */
  private static final class OffHeapVamanaGraph extends VamanaGraph {

    final IndexInput dataIn;
    final long indexOffset;
    final long indexLength;
    final int entryNode;
    final int size;
    final int dimensions;
    final VectorEncoding encoding;
    int arcCount;
    int arcUpTo;
    int arc;
    NodesIterator neighborsIter;
    private final DirectMonotonicReader graphNodeOffsets;
    // Allocated to be M to track the current neighbors being explored
    private final int[] currentNeighborsBuffer;
    private final int vectorSize;

    OffHeapVamanaGraph(FieldEntry entry, IndexInput vectorIndex) throws IOException {
      this.dataIn =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      this.indexOffset = entry.vectorIndexOffset;
      this.indexLength = entry.vectorIndexLength;
      this.entryNode = entry.entryNode;
      this.size = entry.size();
      this.dimensions = entry.dimension;
      this.encoding = entry.vectorEncoding;
      final RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      this.graphNodeOffsets = DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);
      this.currentNeighborsBuffer = new int[entry.M];
      this.vectorSize = this.dimensions * this.encoding.byteSize;
    }

    public void seek(int targetOrd) throws IOException {
      assert targetOrd >= 0;
      // unsafe; no bounds checking

      // seek to the [vector | adjacency list] for this ordinal, then seek past the vector.
      var targetOffset = graphNodeOffsets.get(targetOrd);
      var vectorOffset = this.vectorSize;
      dataIn.seek(targetOffset + vectorOffset);

      arcCount = dataIn.readVInt();
      if (arcCount > 0) {
        currentNeighborsBuffer[0] = dataIn.readVInt();
        for (int i = 1; i < arcCount; i++) {
          currentNeighborsBuffer[i] = currentNeighborsBuffer[i - 1] + dataIn.readVInt();
        }
      }

      neighborsIter = new ArrayNodesIterator(currentNeighborsBuffer, arcCount);

      arc = -1;
      arcUpTo = 0;
    }

    public int size() {
      return size;
    }

    public int nextNeighbor() throws IOException {
      if (arcUpTo >= arcCount) {
        return NO_MORE_DOCS;
      }
      arc = currentNeighborsBuffer[arcUpTo];
      ++arcUpTo;
      return arc;
    }

    public int entryNode() throws IOException {
      return entryNode;
    }

    public NodesIterator getNodes() {
      return new ArrayNodesIterator(size());
    }

    public NodesIterator getNeighbors() {
      return neighborsIter;
    }
  }

  private static class InGraphOffHeapFloatVectorValues {

    final IndexInput dataIn;
    private final int size;
    private final int dimensions;
    private final DirectMonotonicReader graphNodeOffsets;
    private int lastOrd = -1;
    private int doc = -1;
    private final float[] value;

    static InGraphOffHeapFloatVectorValues load(FieldEntry entry, IndexInput vectorIndex)
        throws IOException {
      IndexInput slicedInput =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      DirectMonotonicReader graphNodeOffsets =
          DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);

      return new InGraphOffHeapFloatVectorValues(
          slicedInput, entry.size, entry.dimension, graphNodeOffsets);
    }

    InGraphOffHeapFloatVectorValues(
        IndexInput vectorIndex, int size, int dimensions, DirectMonotonicReader graphNodeOffsets) {
      this.dataIn = vectorIndex;
      this.size = size;
      this.dimensions = dimensions;
      this.graphNodeOffsets = graphNodeOffsets;
      this.value = new float[dimensions];
    }

    public int size() {
      return size;
    }

    public int dimension() {
      return dimensions;
    }

    public float[] vectorValue(int targetOrd) throws IOException {
      if (lastOrd == targetOrd) {
        return value;
      }

      // unsafe; no bounds checking
      long targetOffset = graphNodeOffsets.get(targetOrd);
      dataIn.seek(targetOffset);
      dataIn.readFloats(value, 0, dimensions);
      lastOrd = targetOrd;
      return value;
    }

    public int docID() {
      return doc;
    }
  }

  private static class InGraphOffHeapQuantizedByteVectorValues extends QuantizedByteVectorValues
      implements RandomAccessQuantizedByteVectorValues {

    final IndexInput dataIn;
    private final int size;
    private final int dimensions;
    private final DirectMonotonicReader graphNodeOffsets;
    protected final byte[] binaryValue;
    protected final ByteBuffer byteBuffer;
    private int lastOrd = -1;
    private int doc = -1;
    protected final float[] scoreCorrectionConstant = new float[1];

    static InGraphOffHeapQuantizedByteVectorValues load(FieldEntry entry, IndexInput vectorIndex)
        throws IOException {
      IndexInput slicedInput =
          vectorIndex.slice("graph-data", entry.vectorIndexOffset, entry.vectorIndexLength);
      RandomAccessInput addressesData =
          vectorIndex.randomAccessSlice(entry.offsetsOffset, entry.offsetsLength);
      DirectMonotonicReader graphNodeOffsets =
          DirectMonotonicReader.getInstance(entry.offsetsMeta, addressesData);

      return new InGraphOffHeapQuantizedByteVectorValues(
          slicedInput, entry.size, entry.dimension, graphNodeOffsets);
    }

    InGraphOffHeapQuantizedByteVectorValues(
        IndexInput vectorIndex, int size, int dimensions, DirectMonotonicReader graphNodeOffsets) {
      this.dataIn = vectorIndex;
      this.size = size;
      this.dimensions = dimensions;
      this.graphNodeOffsets = graphNodeOffsets;
      this.byteBuffer = ByteBuffer.allocate(dimensions);
      this.binaryValue = byteBuffer.array();
    }

    @Override
    public int dimension() {
      return dimensions;
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

      // unsafe; no bounds checking
      long targetOffset = graphNodeOffsets.get(targetOrd);
      dataIn.seek(targetOffset);
      dataIn.readBytes(byteBuffer.array(), byteBuffer.arrayOffset(), dimensions);
      dataIn.readFloats(scoreCorrectionConstant, 0, 1);
      lastOrd = targetOrd;
      return binaryValue;
    }

    @Override
    public float getScoreCorrectionConstant() {
      return scoreCorrectionConstant[0];
    }

    @Override
    public RandomAccessQuantizedByteVectorValues copy() throws IOException {
      return new InGraphOffHeapQuantizedByteVectorValues(
          this.dataIn.clone(), this.size, this.dimensions, this.graphNodeOffsets);
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
  }
}
