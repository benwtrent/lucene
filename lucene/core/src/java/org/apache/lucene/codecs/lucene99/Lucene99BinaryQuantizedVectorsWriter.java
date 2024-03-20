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

import static org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.BinaryQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;

/**
 * Writes binary quantized vector values and metadata to index segments.
 *
 * @lucene.experimental
 */
public final class Lucene99BinaryQuantizedVectorsWriter extends FlatVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(Lucene99BinaryQuantizedVectorsWriter.class);

  private final SegmentWriteState segmentWriteState;

  private final List<FieldWriter> fields = new ArrayList<>();
  private final IndexOutput meta, quantizedVectorData;
  private final FlatVectorsWriter rawVectorDelegate;
  private boolean finished;

  public Lucene99BinaryQuantizedVectorsWriter(
      SegmentWriteState state, FlatVectorsWriter rawVectorDelegate) throws IOException {
    segmentWriteState = state;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene99BinaryQuantizedVectorsFormat.META_EXTENSION);

    String quantizedVectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene99BinaryQuantizedVectorsFormat.VECTOR_DATA_EXTENSION);
    this.rawVectorDelegate = rawVectorDelegate;
    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      quantizedVectorData =
          state.directory.createOutput(quantizedVectorDataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          Lucene99BinaryQuantizedVectorsFormat.META_CODEC_NAME,
          Lucene99BinaryQuantizedVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          quantizedVectorData,
          Lucene99BinaryQuantizedVectorsFormat.VECTOR_DATA_CODEC_NAME,
          Lucene99BinaryQuantizedVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public FlatFieldVectorsWriter<?> addField(
      FieldInfo fieldInfo, KnnFieldVectorsWriter<?> indexWriter) throws IOException {
    FieldWriter quantizedWriter = new FieldWriter(fieldInfo, indexWriter);
    fields.add(quantizedWriter);
    indexWriter = quantizedWriter;
    return rawVectorDelegate.addField(fieldInfo, indexWriter);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    rawVectorDelegate.mergeOneField(fieldInfo, mergeState);
    MergedBinaryQuantizedVectorValues byteVectorValues =
        MergedBinaryQuantizedVectorValues.mergeQuantizedByteVectorValues(fieldInfo, mergeState);
    long vectorDataOffset = quantizedVectorData.alignFilePointer(Float.BYTES);
    DocsWithFieldSet docsWithField =
        writeQuantizedVectorData(quantizedVectorData, byteVectorValues);
    long vectorDataLength = quantizedVectorData.getFilePointer() - vectorDataOffset;
    writeMeta(
        fieldInfo,
        segmentWriteState.segmentInfo.maxDoc(),
        vectorDataOffset,
        vectorDataLength,
        docsWithField);
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    long vectorDataOffset = quantizedVectorData.alignFilePointer(Float.BYTES);
    MergedBinaryQuantizedVectorValues byteVectorValues =
        MergedBinaryQuantizedVectorValues.mergeQuantizedByteVectorValues(fieldInfo, mergeState);
    DocsWithFieldSet docsWithField =
        writeQuantizedVectorData(quantizedVectorData, byteVectorValues);
    long vectorDataLength = quantizedVectorData.getFilePointer() - vectorDataOffset;
    writeMeta(
        fieldInfo,
        segmentWriteState.segmentInfo.maxDoc(),
        vectorDataOffset,
        vectorDataLength,
        docsWithField);
    return rawVectorDelegate.mergeOneFieldToIndex(fieldInfo, mergeState);
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    rawVectorDelegate.flush(maxDoc, sortMap);
    for (FieldWriter field : fields) {
      if (sortMap == null) {
        writeField(field, maxDoc);
      } else {
        writeSortingField(field, maxDoc, sortMap);
      }
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    rawVectorDelegate.finish();
    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (quantizedVectorData != null) {
      CodecUtil.writeFooter(quantizedVectorData);
    }
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (FieldWriter field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }

  private void writeField(FieldWriter fieldData, int maxDoc) throws IOException {
    // write vector values
    long vectorDataOffset = quantizedVectorData.alignFilePointer(Float.BYTES);
    writeQuantizedVectors(fieldData);
    long vectorDataLength = quantizedVectorData.getFilePointer() - vectorDataOffset;

    writeMeta(
        fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, fieldData.docsWithField);
  }

  private void writeMeta(
      FieldInfo field,
      int maxDoc,
      long vectorDataOffset,
      long vectorDataLength,
      DocsWithFieldSet docsWithField)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    meta.writeVInt(field.getVectorDimension());
    int count = docsWithField.cardinality();
    meta.writeInt(count);
    // write docIDs
    OrdToDocDISIReaderConfiguration.writeStoredMeta(
        DIRECT_MONOTONIC_BLOCK_SHIFT, meta, quantizedVectorData, count, maxDoc, docsWithField);
  }

  private void writeQuantizedVectors(FieldWriter fieldData) throws IOException {
    BinaryQuantizer scalarQuantizer = new BinaryQuantizer();
    byte[] vector = new byte[fieldData.fieldInfo.getVectorDimension() >> 3];
    for (float[] v : fieldData.floatVectors) {
      scalarQuantizer.quantize(v, vector);
      quantizedVectorData.writeBytes(vector, vector.length);
    }
  }

  private void writeSortingField(FieldWriter fieldData, int maxDoc, Sorter.DocMap sortMap)
      throws IOException {
    final int[] docIdOffsets = new int[sortMap.size()];
    int offset = 1; // 0 means no vector for this (field, document)
    DocIdSetIterator iterator = fieldData.docsWithField.iterator();
    for (int docID = iterator.nextDoc();
        docID != DocIdSetIterator.NO_MORE_DOCS;
        docID = iterator.nextDoc()) {
      int newDocID = sortMap.oldToNew(docID);
      docIdOffsets[newDocID] = offset++;
    }
    DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
    final int[] ordMap = new int[offset - 1]; // new ord to old ord
    int ord = 0;
    int doc = 0;
    for (int docIdOffset : docIdOffsets) {
      if (docIdOffset != 0) {
        ordMap[ord] = docIdOffset - 1;
        newDocsWithField.add(doc);
        ord++;
      }
      doc++;
    }

    // write vector values
    long vectorDataOffset = quantizedVectorData.alignFilePointer(Float.BYTES);
    writeSortedQuantizedVectors(fieldData, ordMap);
    long quantizedVectorLength = quantizedVectorData.getFilePointer() - vectorDataOffset;
    writeMeta(
        fieldData.fieldInfo, maxDoc, vectorDataOffset, quantizedVectorLength, newDocsWithField);
  }

  private void writeSortedQuantizedVectors(FieldWriter fieldData, int[] ordMap) throws IOException {
    BinaryQuantizer scalarQuantizer = new BinaryQuantizer();
    byte[] vector = new byte[fieldData.fieldInfo.getVectorDimension() >> 3];
    for (int ordinal : ordMap) {
      float[] v = fieldData.floatVectors.get(ordinal);
      scalarQuantizer.quantize(v, vector);
      quantizedVectorData.writeBytes(vector, vector.length);
    }
  }

  private static QuantizedVectorsReader getQuantizedKnnVectorsReader(
      KnnVectorsReader vectorsReader, String fieldName) {
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
      vectorsReader = candidateReader.getFieldReader(fieldName);
    }
    if (vectorsReader instanceof QuantizedVectorsReader reader) {
      return reader;
    }
    return null;
  }

  /**
   * Writes the vector values to the output and returns a set of documents that contains vectors.
   */
  public static DocsWithFieldSet writeQuantizedVectorData(
      IndexOutput output, ByteVectorValues quantizedByteVectorValues) throws IOException {
    DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    for (int docV = quantizedByteVectorValues.nextDoc();
        docV != NO_MORE_DOCS;
        docV = quantizedByteVectorValues.nextDoc()) {
      // write vector
      byte[] binaryValue = quantizedByteVectorValues.vectorValue();
      assert binaryValue.length == quantizedByteVectorValues.dimension() >> 3
          : "dim=" + quantizedByteVectorValues.dimension() + " len=" + binaryValue.length;
      output.writeBytes(binaryValue, binaryValue.length);
      docsWithField.add(docV);
    }
    return docsWithField;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, quantizedVectorData, rawVectorDelegate);
  }

  static class FieldWriter extends FlatFieldVectorsWriter<float[]> {
    private static final long SHALLOW_SIZE = shallowSizeOfInstance(FieldWriter.class);
    private final List<float[]> floatVectors;
    private final FieldInfo fieldInfo;
    private final DocsWithFieldSet docsWithField;

    @SuppressWarnings("unchecked")
    FieldWriter(FieldInfo fieldInfo, KnnFieldVectorsWriter<?> indexWriter) {
      super((KnnFieldVectorsWriter<float[]>) indexWriter);
      this.fieldInfo = fieldInfo;
      this.floatVectors = new ArrayList<>();
      this.docsWithField = new DocsWithFieldSet();
    }

    @Override
    public long ramBytesUsed() {
      long size = SHALLOW_SIZE;
      if (indexingDelegate != null) {
        size += indexingDelegate.ramBytesUsed();
      }
      if (floatVectors.size() == 0) return size;
      return size + (long) floatVectors.size() * RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    }

    @Override
    public void addValue(int docID, float[] vectorValue) throws IOException {
      docsWithField.add(docID);
      floatVectors.add(vectorValue);
      if (indexingDelegate != null) {
        indexingDelegate.addValue(docID, vectorValue);
      }
    }

    @Override
    public float[] copyValue(float[] vectorValue) {
      throw new UnsupportedOperationException();
    }
  }

  private static class BinaryQuantizedByteVectorValueSub extends DocIDMerger.Sub {
    private final ByteVectorValues values;

    BinaryQuantizedByteVectorValueSub(MergeState.DocMap docMap, QuantizedByteVectorValues values) {
      super(docMap);
      this.values = values;
      assert values.docID() == -1;
    }

    @Override
    public int nextDoc() throws IOException {
      return values.nextDoc();
    }
  }

  /** Returns a merged view over all the segment's {@link ByteVectorValues}. */
  static class MergedBinaryQuantizedVectorValues extends ByteVectorValues {
    public static MergedBinaryQuantizedVectorValues mergeQuantizedByteVectorValues(
        FieldInfo fieldInfo, MergeState mergeState) throws IOException {
      assert fieldInfo != null && fieldInfo.hasVectorValues();

      List<BinaryQuantizedByteVectorValueSub> subs = new ArrayList<>();
      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        if (mergeState.knnVectorsReaders[i] != null) {
          QuantizedVectorsReader reader =
              getQuantizedKnnVectorsReader(mergeState.knnVectorsReaders[i], fieldInfo.name);
          final BinaryQuantizedByteVectorValueSub sub;
          // Either our quantization parameters are way different than the merged ones
          // Or we have never been quantized.
          if (reader == null) {
            sub =
                new BinaryQuantizedByteVectorValueSub(
                    mergeState.docMaps[i],
                    new QuantizedFloatVectorValues(
                        mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name)));
          } else {
            sub =
                new BinaryQuantizedByteVectorValueSub(
                    mergeState.docMaps[i], reader.getQuantizedVectorValues(fieldInfo.name));
          }
          subs.add(sub);
        }
      }
      return new MergedBinaryQuantizedVectorValues(subs, mergeState);
    }

    private final List<BinaryQuantizedByteVectorValueSub> subs;
    private final DocIDMerger<BinaryQuantizedByteVectorValueSub> docIdMerger;
    private final int size;

    private int docId;
    private BinaryQuantizedByteVectorValueSub current;

    private MergedBinaryQuantizedVectorValues(
        List<BinaryQuantizedByteVectorValueSub> subs, MergeState mergeState) throws IOException {
      this.subs = subs;
      docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
      int totalSize = 0;
      for (BinaryQuantizedByteVectorValueSub sub : subs) {
        totalSize += sub.values.size();
      }
      size = totalSize;
      docId = -1;
    }

    @Override
    public byte[] vectorValue() throws IOException {
      return current.values.vectorValue();
    }

    @Override
    public int docID() {
      return docId;
    }

    @Override
    public int nextDoc() throws IOException {
      current = docIdMerger.next();
      if (current == null) {
        docId = NO_MORE_DOCS;
      } else {
        docId = current.mappedDocID;
      }
      return docId;
    }

    @Override
    public int advance(int target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return subs.get(0).values.dimension();
    }
  }

  private static class QuantizedFloatVectorValues extends QuantizedByteVectorValues {
    private final FloatVectorValues values;
    private final BinaryQuantizer quantizer = new BinaryQuantizer();
    private final byte[] quantizedVector;

    public QuantizedFloatVectorValues(FloatVectorValues values) {
      this.values = values;
      this.quantizedVector = new byte[values.dimension()];
    }

    @Override
    public float getScoreCorrectionConstant() {
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

    @Override
    public byte[] vectorValue() throws IOException {
      return quantizedVector;
    }

    @Override
    public int docID() {
      return values.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      int doc = values.nextDoc();
      if (doc != NO_MORE_DOCS) {
        quantize();
      }
      return doc;
    }

    @Override
    public int advance(int target) throws IOException {
      int doc = values.advance(target);
      if (doc != NO_MORE_DOCS) {
        quantize();
      }
      return doc;
    }

    private void quantize() throws IOException {
      quantizer.quantize(values.vectorValue(), quantizedVector);
    }
  }
}
