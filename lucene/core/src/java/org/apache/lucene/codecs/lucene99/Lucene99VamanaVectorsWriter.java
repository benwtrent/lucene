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

import static org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsWriter.mergeAndRecalculateQuantiles;
import static org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsWriter.writeQuantizedVectorData;
import static org.apache.lucene.codecs.lucene99.Lucene99VamanaVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.apache.lucene.util.vamana.VamanaGraph.NodesIterator.getSortedNodes;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.ScalarQuantizer;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.packed.DirectMonotonicWriter;
import org.apache.lucene.util.vamana.ConcurrentVamanaMerger;
import org.apache.lucene.util.vamana.IncrementalVamanaGraphMerger;
import org.apache.lucene.util.vamana.OnHeapVamanaGraph;
import org.apache.lucene.util.vamana.VamanaGraph;
import org.apache.lucene.util.vamana.VamanaGraphBuilder;
import org.apache.lucene.util.vamana.VamanaGraphMerger;

/**
 * Writes vector values and knn graphs to index segments.
 *
 * @lucene.experimental
 */
public final class Lucene99VamanaVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      RamUsageEstimator.shallowSizeOfInstance(Lucene99VamanaVectorsWriter.class);
  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta, vectorIndex;
  private final int M;
  private final int beamWidth;
  private final FlatVectorsWriter flatVectorWriter;
  private final List<FieldWriter<?>> fields = new ArrayList<>();
  private final int numMergeWorkers;
  private final TaskExecutor mergeExec;
  private boolean finished;

  Lucene99VamanaVectorsWriter(
      SegmentWriteState state,
      int M,
      int beamWidth,
      FlatVectorsWriter flatVectorWriter,
      int numMergeWorkers,
      TaskExecutor mergeExec)
      throws IOException {
    this.M = M;
    this.flatVectorWriter = flatVectorWriter;
    this.beamWidth = beamWidth;
    segmentWriteState = state;
    this.numMergeWorkers = numMergeWorkers;
    this.mergeExec = mergeExec;

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene99VamanaVectorsFormat.META_EXTENSION);

    String indexDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene99VamanaVectorsFormat.VECTOR_INDEX_EXTENSION);

    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorIndex = state.directory.createOutput(indexDataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          Lucene99VamanaVectorsFormat.META_CODEC_NAME,
          Lucene99VamanaVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorIndex,
          Lucene99VamanaVectorsFormat.VECTOR_INDEX_CODEC_NAME,
          Lucene99VamanaVectorsFormat.VERSION_CURRENT,
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
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    FieldWriter<?> newField =
        FieldWriter.create(fieldInfo, M, beamWidth, 1.2f, segmentWriteState.infoStream);
    fields.add(newField);
    return flatVectorWriter.addField(fieldInfo, newField);
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    flatVectorWriter.flush(maxDoc, sortMap);
    for (FieldWriter<?> field : fields) {
      field.finish();
      if (sortMap == null) {
        writeField(field);
      } else {
        writeSortingField(field, sortMap);
      }
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorWriter.finish();

    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorIndex != null) {
      CodecUtil.writeFooter(vectorIndex);
    }
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    total += flatVectorWriter.ramBytesUsed();
    for (FieldWriter<?> field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }

  private void writeField(FieldWriter<?> fieldData) throws IOException {
    // write graph
    long vectorIndexOffset = vectorIndex.getFilePointer();
    int[] graphLevelNodeOffsets = writeGraph(fieldData);
    long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;

    writeMeta(
        fieldData.fieldInfo,
        vectorIndexOffset,
        vectorIndexLength,
        fieldData.docsWithField.cardinality(),
        fieldData.getGraph(),
        fieldData.quantizer,
        graphLevelNodeOffsets);
  }

  private void writeSortingField(FieldWriter<?> fieldData, Sorter.DocMap sortMap)
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
    final int[] oldOrdMap = new int[offset - 1]; // old ord to new ord
    int ord = 0;
    int doc = 0;
    for (int docIdOffset : docIdOffsets) {
      if (docIdOffset != 0) {
        ordMap[ord] = docIdOffset - 1;
        oldOrdMap[docIdOffset - 1] = ord;
        newDocsWithField.add(doc);
        ord++;
      }
      doc++;
    }
    // write graph
    long vectorIndexOffset = vectorIndex.getFilePointer();
    VamanaGraph graph = fieldData.getGraph();
    int[] nodeOffsets = graph == null ? new int[0] : new int[graph.size()];
    VamanaGraph mockGraph = reconstructAndWriteGraph(fieldData, ordMap, oldOrdMap, nodeOffsets);
    long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;

    writeMeta(
        fieldData.fieldInfo,
        vectorIndexOffset,
        vectorIndexLength,
        fieldData.docsWithField.cardinality(),
        mockGraph,
        fieldData.quantizer,
        nodeOffsets);
  }

  private VamanaGraph reconstructAndWriteGraph(
      FieldWriter<?> fieldData, int[] newToOldMap, int[] oldToNewMap, int[] nodeOffsets)
      throws IOException {
    List<?> vectors = fieldData.vectors;
    OnHeapVamanaGraph graph = fieldData.getGraph();
    if (fieldData.builtGraph == null) {
      return null;
    }
    VectorEncoding encoding = fieldData.fieldInfo.getVectorEncoding();
    ByteBuffer quantizationOffsetBuffer =
        ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    ScalarQuantizer quantizer = fieldData.quantizer;
    byte[] quantizedVector = new byte[fieldData.dim];
    float[] normalizeCopy =
        fieldData.fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE
            ? new float[fieldData.dim]
            : null;

    int maxOrd = graph.size();
    VamanaGraph.NodesIterator nodes = graph.getNodes();
    while (nodes.hasNext()) {
      long offset = vectorIndex.getFilePointer();

      int node = nodes.nextInt();

      switch (encoding) {
        case BYTE -> {
          byte[] v = (byte[]) vectors.get(node);
          vectorIndex.writeBytes(v, v.length);
        }
        case FLOAT32 -> {
          assert fieldData.quantizer != null;
          float[] vector = (float[]) fieldData.vectors.get(node);
          if (fieldData.fieldInfo.getVectorSimilarityFunction()
              == VectorSimilarityFunction.COSINE) {
            System.arraycopy(vector, 0, normalizeCopy, 0, normalizeCopy.length);
            VectorUtil.l2normalize(normalizeCopy);
          }
          float offsetCorrection =
              quantizer.quantize(
                  normalizeCopy != null ? normalizeCopy : vector,
                  quantizedVector,
                  fieldData.fieldInfo.getVectorSimilarityFunction());
          vectorIndex.writeBytes(quantizedVector, quantizedVector.length);
          quantizationOffsetBuffer.putFloat(offsetCorrection);
          vectorIndex.writeBytes(
              quantizationOffsetBuffer.array(), quantizationOffsetBuffer.array().length);
          quantizationOffsetBuffer.rewind();
        }
      }

      NeighborArray neighbors = graph.getNeighbors(newToOldMap[node]);
      reconstructAndWriteNeighbours(neighbors, oldToNewMap, maxOrd);

      if (encoding == VectorEncoding.FLOAT32) {
        vectorIndex.alignFilePointer(Float.BYTES);
      }

      nodeOffsets[node] = Math.toIntExact(vectorIndex.getFilePointer() - offset);
    }

    return new VamanaGraph() {
      @Override
      public int nextNeighbor() {
        throw new UnsupportedOperationException("Not supported on a mock graph");
      }

      @Override
      public void seek(int target) {
        throw new UnsupportedOperationException("Not supported on a mock graph");
      }

      @Override
      public int size() {
        return graph.size();
      }

      @Override
      public int entryNode() throws IOException {
        return oldToNewMap[graph.entryNode()];
      }

      @Override
      public NodesIterator getNeighbors() {
        return graph.getNeighbors();
      }

      @Override
      public NodesIterator getNodes() throws IOException {
        return graph.getNodes();
      }
    };
  }

  private void reconstructAndWriteNeighbours(NeighborArray neighbors, int[] oldToNewMap, int maxOrd)
      throws IOException {
    int size = neighbors.size();
    vectorIndex.writeVInt(size);

    // Destructively modify; it's ok we are discarding it after this
    int[] nnodes = neighbors.nodes();
    for (int i = 0; i < size; i++) {
      nnodes[i] = oldToNewMap[nnodes[i]];
    }
    Arrays.sort(nnodes, 0, size);
    // Now that we have sorted, do delta encoding to minimize the required bits to store the
    // information
    for (int i = size - 1; i > 0; --i) {
      assert nnodes[i] < maxOrd : "node too large: " + nnodes[i] + ">=" + maxOrd;
      nnodes[i] -= nnodes[i - 1];
    }
    for (int i = 0; i < size; i++) {
      vectorIndex.writeVInt(nnodes[i]);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    // TODO mergeToIndex if its a byte field
    flatVectorWriter.mergeOneField(fieldInfo, mergeState);
    IndexOutput tempQuantizedVectorData =
        segmentWriteState.directory.createTempOutput(
            vectorIndex.getName() + "_q8", "temp", segmentWriteState.context);
    IndexInput quantizationDataInput = null;
    boolean success = false;
    try {
      // merge quantizaiton actions
      ScalarQuantizer mergedQuantizationState =
          mergeAndRecalculateQuantiles(
              mergeState,
              fieldInfo,
              Lucene99ScalarQuantizedVectorsFormat.calculateDefaultConfidenceInterval(
                  fieldInfo.getVectorDimension()));
      Lucene99ScalarQuantizedVectorsWriter.MergedQuantizedVectorValues byteVectorValues =
          Lucene99ScalarQuantizedVectorsWriter.MergedQuantizedVectorValues
              .mergeQuantizedByteVectorValues(fieldInfo, mergeState, mergedQuantizationState);
      DocsWithFieldSet docsWithField =
          writeQuantizedVectorData(tempQuantizedVectorData, byteVectorValues);
      CodecUtil.writeFooter(tempQuantizedVectorData);
      IOUtils.close(tempQuantizedVectorData);
      quantizationDataInput =
          segmentWriteState.directory.openInput(
              tempQuantizedVectorData.getName(), segmentWriteState.context);
      CodecUtil.retrieveChecksum(quantizationDataInput);
      RandomVectorScorerSupplier scorerSupplier =
          new ScalarQuantizedRandomVectorScorerSupplier(
              fieldInfo.getVectorSimilarityFunction(),
              mergedQuantizationState,
              new OffHeapQuantizedByteVectorValues.DenseOffHeapVectorValues(
                  fieldInfo.getVectorDimension(),
                  docsWithField.cardinality(),
                  quantizationDataInput));

      long vectorIndexOffset = vectorIndex.getFilePointer();
      // build the graph using the temporary vector data
      // we use Lucene99HnswVectorsReader.DenseOffHeapVectorValues for the graph construction
      // doesn't need to know docIds
      // TODO: separate random access vector values from DocIdSetIterator?
      OnHeapVamanaGraph graph = null;
      int[] vectorIndexNodeOffsets = null;
      if (docsWithField.cardinality() > 0) {
        // build graph
        VamanaGraphMerger merger = createGraphMerger(fieldInfo, scorerSupplier);
        for (int i = 0; i < mergeState.liveDocs.length; i++) {
          merger.addReader(
              mergeState.knnVectorsReaders[i], mergeState.docMaps[i], mergeState.liveDocs[i]);
        }
        graph =
            merger.merge(
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState),
              segmentWriteState.infoStream,
              docsWithField.cardinality()
            );
        vectorIndexNodeOffsets =
            writeMergedGraph(
                graph,
                fieldInfo.getVectorEncoding(),
                fieldInfo.getVectorSimilarityFunction(),
                fieldInfo.getVectorDimension(),
                mergedQuantizationState,
                new OffHeapQuantizedByteVectorValues.DenseOffHeapVectorValues(
                fieldInfo.getVectorDimension(),
                docsWithField.cardinality(),
                quantizationDataInput));
      }
      long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;
      writeMeta(
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          docsWithField.cardinality(),
          graph,
          mergedQuantizationState,
          vectorIndexNodeOffsets);
      success = true;
    } finally {
      if (success) {
        IOUtils.close(quantizationDataInput);
        segmentWriteState.directory.deleteFile(tempQuantizedVectorData.getName());
      } else {
        IOUtils.closeWhileHandlingException(tempQuantizedVectorData, quantizationDataInput);
        IOUtils.deleteFilesIgnoringExceptions(
            segmentWriteState.directory, tempQuantizedVectorData.getName());
      }
    }
  }

  private VamanaGraphMerger createGraphMerger(
      FieldInfo fieldInfo, RandomVectorScorerSupplier scorerSupplier) {
    if (mergeExec != null) {
      return new ConcurrentVamanaMerger(
          fieldInfo, scorerSupplier, M, beamWidth, 1.2f, mergeExec, numMergeWorkers);
    }
    return new IncrementalVamanaGraphMerger(fieldInfo, scorerSupplier, M, beamWidth, 1.2f);
  }

  private int[] writeMergedGraph(
      OnHeapVamanaGraph graph,
      VectorEncoding encoding,
      VectorSimilarityFunction similarityFunction,
      int dimensions,
      ScalarQuantizer quantizer,
      DocIdSetIterator vectors)
      throws IOException {
    boolean quantized = quantizer != null;
    ByteBuffer vectorBuffer =
        encoding == VectorEncoding.FLOAT32
            ? ByteBuffer.allocate(dimensions * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN)
            : null;
    ByteBuffer quantizationOffsetBuffer =
        quantized ? ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN) : null;
    float[] normalizeCopy =
        quantized && similarityFunction == VectorSimilarityFunction.COSINE
            ? new float[dimensions]
            : null;

    int[] sortedNodes = getSortedNodes(graph.getNodes());
    int[] offsets = new int[sortedNodes.length];
    int nodeOffsetId = 0;
    QuantizedByteVectorValues quantizedVectors = (QuantizedByteVectorValues) vectors;

    for (int node : sortedNodes) {
      long offsetStart = vectorIndex.getFilePointer();
      int docId = vectors.nextDoc();
      if (docId != node) {
        throw new IllegalStateException("docId=" + docId + " node=" + node);
      }
      assert docId == node;
      // Write the full fidelity vector
      switch (encoding) {
        case BYTE -> {
          byte[] v = ((ByteVectorValues) vectors).vectorValue();
          vectorIndex.writeBytes(v, v.length);
        }
        case FLOAT32 -> {
          byte[] quantizedVector = quantizedVectors.vectorValue();
          float offsetCorrection = quantizedVectors.getScoreCorrectionConstant();
          vectorIndex.writeBytes(quantizedVector, quantizedVector.length);
          quantizationOffsetBuffer.putFloat(offsetCorrection);
          vectorIndex.writeBytes(
              quantizationOffsetBuffer.array(), quantizationOffsetBuffer.array().length);
          quantizationOffsetBuffer.rewind();
        }
      }

      NeighborArray neighbors = graph.getNeighbors(node);
      int size = neighbors.size();

      // Write size in VInt as the neighbors list is typically small
      vectorIndex.writeVInt(size);

      // Encode neighbors as vints.
      int[] nnodes = neighbors.nodes();
      Arrays.sort(nnodes, 0, size);

      // Convert neighbors to their deltas from the previous neighbor.
      for (int i = size - 1; i > 0; --i) {
        nnodes[i] -= nnodes[i - 1];
      }
      for (int i = 0; i < size; i++) {
        vectorIndex.writeVInt(nnodes[i]);
      }

      if (encoding == VectorEncoding.FLOAT32 && !quantized) {
        vectorIndex.alignFilePointer(Float.BYTES);
      }

      var offset = Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
      offsets[nodeOffsetId++] = offset;
    }

    return offsets;
  }

  /**
   * @throws IOException if writing to vectorIndex fails
   */
  private int[] writeGraph(FieldWriter<?> fieldData) throws IOException {
    OnHeapVamanaGraph graph = fieldData.getGraph();
    if (graph == null) {
      return new int[0];
    }

    VectorEncoding encoding = fieldData.fieldInfo.getVectorEncoding();
    ByteBuffer quantizationOffsetBuffer =
        ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    ScalarQuantizer quantizer = fieldData.quantizer;
    byte[] quantizedVector = new byte[fieldData.dim];
    float[] normalizeCopy =
        fieldData.fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE
            ? new float[fieldData.dim]
            : null;

    int[] sortedNodes = getSortedNodes(graph.getNodes());
    int[] offsets = new int[sortedNodes.length];
    int nodeOffsetId = 0;

    for (int node : sortedNodes) {
      long offsetStart = vectorIndex.getFilePointer();

      // Write the full fidelity vector
      if (true) {
        switch (encoding) {
          case BYTE -> {
            byte[] v = (byte[]) fieldData.vectors.get(node);
            vectorIndex.writeBytes(v, v.length);
          }
          case FLOAT32 -> {
            float[] vector = (float[]) fieldData.vectors.get(node);
            if (fieldData.fieldInfo.getVectorSimilarityFunction()
                == VectorSimilarityFunction.COSINE) {
              System.arraycopy(vector, 0, normalizeCopy, 0, normalizeCopy.length);
              VectorUtil.l2normalize(normalizeCopy);
            }
            float offsetCorrection =
                quantizer.quantize(
                    normalizeCopy != null ? normalizeCopy : vector,
                    quantizedVector,
                    fieldData.fieldInfo.getVectorSimilarityFunction());
            vectorIndex.writeBytes(quantizedVector, quantizedVector.length);
            quantizationOffsetBuffer.putFloat(offsetCorrection);
            vectorIndex.writeBytes(
                quantizationOffsetBuffer.array(), quantizationOffsetBuffer.array().length);
            quantizationOffsetBuffer.rewind();
          }
        }
      }

      NeighborArray neighbors = graph.getNeighbors(node);
      int size = neighbors.size();

      // Write size in VInt as the neighbors list is typically small
      vectorIndex.writeVInt(size);

      // Encode neighbors as vints.
      int[] nnodes = neighbors.nodes();
      Arrays.sort(nnodes, 0, size);

      // Convert neighbors to their deltas from the previous neighbor.
      for (int i = size - 1; i > 0; --i) {
        nnodes[i] -= nnodes[i - 1];
      }
      for (int i = 0; i < size; i++) {
        vectorIndex.writeVInt(nnodes[i]);
      }

      if (encoding == VectorEncoding.FLOAT32) {
        vectorIndex.alignFilePointer(Float.BYTES);
      }

      var offset = Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
      offsets[nodeOffsetId++] = offset;
    }

    return offsets;
  }

  private void writeMeta(
      FieldInfo field,
      long vectorIndexOffset,
      long vectorIndexLength,
      int count,
      VamanaGraph graph,
      ScalarQuantizer quantizer,
      int[] graphNodeOffsets)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeByte((byte) 1);
    assert quantizer != null;
    meta.writeInt(Float.floatToIntBits(quantizer.getLowerQuantile()));
    meta.writeInt(Float.floatToIntBits(quantizer.getUpperQuantile()));
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());

    // write docIDs
    meta.writeInt(count);

    meta.writeVInt(M);
    // write graph nodes on each level
    if (graph == null) {
      meta.writeVInt(0);
    } else {
      meta.writeVInt(graph.entryNode());
      long start = vectorIndex.getFilePointer();
      meta.writeLong(start);
      meta.writeVInt(DIRECT_MONOTONIC_BLOCK_SHIFT);
      final DirectMonotonicWriter memoryOffsetsWriter =
          DirectMonotonicWriter.getInstance(
              meta, vectorIndex, graph.size(), DIRECT_MONOTONIC_BLOCK_SHIFT);
      long cumulativeOffsetSum = 0;
      for (int v : graphNodeOffsets) {
        memoryOffsetsWriter.add(cumulativeOffsetSum);
        cumulativeOffsetSum += v;
      }
      memoryOffsetsWriter.finish();
      meta.writeLong(vectorIndex.getFilePointer() - start);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorIndex, flatVectorWriter);
  }

  private static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
    private final FieldInfo fieldInfo;
    private final int dim;
    private final DocsWithFieldSet docsWithField;
    private final List<T> vectors;
    private final VamanaGraphBuilder vamanaGraphBuilder;
    private OnHeapVamanaGraph builtGraph;
    ScalarQuantizer quantizer;

    private int lastDocID = -1;
    private int node = 0;

    static FieldWriter<?> create(
        FieldInfo fieldInfo, int M, int beamWidth, float alpha, InfoStream infoStream)
        throws IOException {
      int dim = fieldInfo.getVectorDimension();
      return switch (fieldInfo.getVectorEncoding()) {
        case BYTE -> new FieldWriter<byte[]>(fieldInfo, M, beamWidth, alpha, infoStream) {
          @Override
          public byte[] copyValue(byte[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, dim);
          }
        };
        case FLOAT32 -> new FieldWriter<float[]>(fieldInfo, M, beamWidth, alpha, infoStream) {
          @Override
          public float[] copyValue(float[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, dim);
          }
        };
      };
    }

    @SuppressWarnings("unchecked")
    FieldWriter(FieldInfo fieldInfo, int M, int beamWidth, float alpha, InfoStream infoStream)
        throws IOException {
      this.fieldInfo = fieldInfo;
      this.dim = fieldInfo.getVectorDimension();
      this.docsWithField = new DocsWithFieldSet();
      vectors = new ArrayList<>();
      Lucene99HnswVectorsWriter.RAVectorValues<T> raVectors =
          new Lucene99HnswVectorsWriter.RAVectorValues<>(vectors, dim);
      RandomVectorScorerSupplier scorerSupplier =
          switch (fieldInfo.getVectorEncoding()) {
            case BYTE -> RandomVectorScorerSupplier.createBytes(
                (RandomAccessVectorValues<byte[]>) raVectors,
                fieldInfo.getVectorSimilarityFunction());
            case FLOAT32 -> RandomVectorScorerSupplier.createFloats(
                (RandomAccessVectorValues<float[]>) raVectors,
                fieldInfo.getVectorSimilarityFunction());
          };
      vamanaGraphBuilder = VamanaGraphBuilder.create(scorerSupplier, M, beamWidth, alpha);
      vamanaGraphBuilder.setInfoStream(infoStream);
    }

    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
      if (docID == lastDocID) {
        throw new IllegalArgumentException(
            "VectorValuesField \""
                + fieldInfo.name
                + "\" appears more than once in this document (only one value is allowed per field)");
      }
      assert docID > lastDocID;
      T copy = copyValue(vectorValue);
      docsWithField.add(docID);
      vectors.add(copy);
      vamanaGraphBuilder.addGraphNode(node);
      node++;
      lastDocID = docID;
    }

    @Override
    public T copyValue(T vectorValue) {
      return null;
    }

    OnHeapVamanaGraph getGraph() {
      return builtGraph;
    }

    @SuppressWarnings("unchecked")
    void finish() throws IOException {
      vamanaGraphBuilder.finish();
      if (vectors.size() > 0) {
        builtGraph = vamanaGraphBuilder.getGraph();
        quantizer =
            ScalarQuantizer.fromVectors(
                new Lucene99ScalarQuantizedVectorsWriter.FloatVectorWrapper(
                    (List<float[]>) vectors,
                    fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE),
                Lucene99ScalarQuantizedVectorsFormat.calculateDefaultConfidenceInterval(dim));
      }
    }

    @Override
    public long ramBytesUsed() {
      if (vectors.size() == 0) {
        return 0;
      }
      return docsWithField.ramBytesUsed()
          + (long) vectors.size()
              * (RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
          + (long) vectors.size()
              * fieldInfo.getVectorDimension()
              * fieldInfo.getVectorEncoding().byteSize
          + vamanaGraphBuilder.getGraph().ramBytesUsed();
    }
  }
}
