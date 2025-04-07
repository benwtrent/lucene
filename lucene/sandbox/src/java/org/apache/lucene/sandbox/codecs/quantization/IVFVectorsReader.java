/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;

import java.io.IOException;
import java.util.function.IntPredicate;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.sandbox.search.knn.IVFKnnSearchStrategy;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.NeighborQueue;

/**
 * @lucene.experimental
 */
public abstract class IVFVectorsReader extends KnnVectorsReader {

  private final IndexInput ivfCentroids, ivfClusters;
  private final SegmentReadState state;
  private final FieldInfos fieldInfos;
  protected final IntObjectHashMap<FieldEntry> fields;
  private final FlatVectorsReader rawVectorsReader;

  protected IVFVectorsReader(SegmentReadState state, FlatVectorsReader rawVectorsReader)
      throws IOException {
    this.state = state;
    this.fieldInfos = state.fieldInfos;
    this.rawVectorsReader = rawVectorsReader;
    this.fields = new IntObjectHashMap<>();
    String meta =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.IVF_META_EXTENSION);

    int versionMeta = -1;
    boolean success = false;
    try (ChecksumIndexInput ivfMeta = state.directory.openChecksumInput(meta)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                ivfMeta,
                IVFVectorsFormat.NAME,
                IVFVectorsFormat.VERSION_START,
                IVFVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(ivfMeta);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(ivfMeta, priorE);
      }
      ivfCentroids =
          openDataInput(
              state,
              versionMeta,
              IVFVectorsFormat.CENTROID_EXTENSION,
              IVFVectorsFormat.NAME,
              state.context);
      ivfClusters =
          openDataInput(
              state,
              versionMeta,
              IVFVectorsFormat.CLUSTER_EXTENSION,
              IVFVectorsFormat.NAME,
              state.context);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  protected abstract IVFUtils.CentroidQueryScorer getCentroidScorer(
      FieldInfo fieldInfo,
      int numCentroids,
      IndexInput centroids,
      float[] target,
      IndexInput clusters)
      throws IOException;

  protected abstract FloatVectorValues getCentroids(
      IndexInput indexInput, int numCentroids, FieldInfo info) throws IOException;

  public FloatVectorValues getCentroids(FieldInfo fieldInfo) throws IOException {
    FieldEntry entry = fields.get(fieldInfo.number);
    if (entry == null) {
      return null;
    }
    return getCentroids(
        entry.centroidSlice(ivfCentroids), entry.postingListOffsets.length, fieldInfo);
  }

  int centroidSize(String fieldName, int centroidOrdinal) throws IOException {
    FieldInfo fieldInfo = state.fieldInfos.fieldInfo(fieldName);
    FieldEntry entry = fields.get(fieldInfo.number);
    ivfClusters.seek(entry.postingListOffsets[centroidOrdinal]);
    return ivfClusters.readVInt();
  }

  private static IndexInput openDataInput(
      SegmentReadState state,
      int versionMeta,
      String fileExtension,
      String codecName,
      IOContext context)
      throws IOException {
    final String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    final IndexInput in = state.directory.openInput(fileName, context);
    boolean success = false;
    try {
      final int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              codecName,
              IVFVectorsFormat.VERSION_START,
              IVFVectorsFormat.VERSION_CURRENT,
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

  private void readFields(ChecksumIndexInput meta) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      final FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      fields.put(info.number, readField(meta, info));
    }
  }

  private FieldEntry readField(IndexInput input, FieldInfo info) throws IOException {
    final VectorEncoding vectorEncoding = readVectorEncoding(input);
    final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    final long centroidOffset = input.readLong();
    final long centroidLength = input.readLong();
    final int numPostingLists = input.readVInt();
    final long[] postingListOffsets = new long[numPostingLists];
    for (int i = 0; i < numPostingLists; i++) {
      postingListOffsets[i] = input.readLong();
    }
    final float[] globalCentroid = new float[info.getVectorDimension()];
    float globalCentroidDp = 0;
    if (numPostingLists > 0) {
      input.readFloats(globalCentroid, 0, globalCentroid.length);
      globalCentroidDp = Float.intBitsToFloat(input.readInt());
    }
    if (similarityFunction != info.getVectorSimilarityFunction()) {
      throw new IllegalStateException(
          "Inconsistent vector similarity function for field=\""
              + info.name
              + "\"; "
              + similarityFunction
              + " != "
              + info.getVectorSimilarityFunction());
    }
    return new FieldEntry(
        similarityFunction,
        vectorEncoding,
        centroidOffset,
        centroidLength,
        postingListOffsets,
        globalCentroid,
        globalCentroidDp);
  }

  private static VectorSimilarityFunction readSimilarityFunction(DataInput input)
      throws IOException {
    final int i = input.readInt();
    if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
      throw new IllegalArgumentException("invalid distance function: " + i);
    }
    return SIMILARITY_FUNCTIONS.get(i);
  }

  private static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
    final int encodingId = input.readInt();
    if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
      throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
    }
    return VectorEncoding.values()[encodingId];
  }

  @Override
  public final void checkIntegrity() throws IOException {
    rawVectorsReader.checkIntegrity();
    CodecUtil.checksumEntireFile(ivfCentroids);
    CodecUtil.checksumEntireFile(ivfClusters);
  }

  @Override
  public final FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return rawVectorsReader.getFloatVectorValues(field);
  }

  @Override
  public final ByteVectorValues getByteVectorValues(String field) throws IOException {
    return rawVectorsReader.getByteVectorValues(field);
  }

  protected float[] getGlobalCentroid(FieldInfo info) {
    if (info == null || info.getVectorEncoding().equals(VectorEncoding.BYTE)) {
      return null;
    }
    FieldEntry entry = fields.get(info.number);
    if (entry == null) {
      return null;
    }
    return entry.globalCentroid();
  }

  @Override
  public final void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
    if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
      rawVectorsReader.search(field, target, knnCollector, acceptDocs);
      return;
    }
    int nProbe = -1;
    if (knnCollector.getSearchStrategy() instanceof IVFKnnSearchStrategy ivfStrategy) {
      nProbe = ivfStrategy.getNProbe();
    }
    BitSet visitedDocs = new FixedBitSet(state.segmentInfo.maxDoc() + 1);
    // TODO can we make a conjunction between idSetIterator and the acceptDocs?
    IntPredicate needsScoring =
        docId -> {
          if (acceptDocs != null && acceptDocs.get(docId) == false) {
            return false;
          }
          return visitedDocs.getAndSet(docId) == false;
        };

    FieldEntry entry = fields.get(fieldInfo.number);
    IVFUtils.CentroidQueryScorer centroidQueryScorer =
        getCentroidScorer(
            fieldInfo,
            entry.postingListOffsets.length,
            entry.centroidSlice(ivfCentroids),
            target,
            ivfClusters);
    int centroidsToSearch = nProbe;
    if (centroidsToSearch <= 0) {
      centroidsToSearch = Math.max(((knnCollector.k() * 300) / 1_000), 1);
    }
    final NeighborQueue centroidQueue =
        scorePostingLists(fieldInfo, knnCollector, centroidQueryScorer, nProbe);
    IVFUtils.PostingVisitor scorer =
        getPostingVisitor(fieldInfo, ivfClusters, target, needsScoring);
    int centroidsVisited = 0;
    long expectedDocs = 0;
    long actualDocs = 0;
    // initially we visit only the "centroids to search"
    while (centroidQueue.size() > 0 && centroidsVisited < centroidsToSearch) {
      ++centroidsVisited;
      // todo do we actually need to know the score???
      int centroidOrdinal = centroidQueue.pop();
      // todo do we need direct access to the raw centroid???
      expectedDocs +=
          scorer.resetPostingsScorer(
              centroidOrdinal, centroidQueryScorer.centroid(centroidOrdinal));
      actualDocs += scorer.visit(knnCollector);
      if (knnCollector.earlyTerminated()) {
        return;
      }
    }
    // if we are using a filtered search, we need to account for the documents that were filtered
    // so continue exploring past centroidsToSearch until we reach the expected number of documents
    // TODO, can we pick something smarter than 0.9? Something related to average posting list size?
    float expectedScored = expectedDocs * 0.9f;
    while (acceptDocs != null && centroidQueue.size() > 0 && actualDocs < expectedScored) {
      int centroidOrdinal = centroidQueue.pop();
      scorer.resetPostingsScorer(centroidOrdinal, centroidQueryScorer.centroid(centroidOrdinal));
      actualDocs += scorer.visit(knnCollector);
      if (knnCollector.earlyTerminated()) {
        return;
      }
    }
  }

  @Override
  public final void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
    final ByteVectorValues values = rawVectorsReader.getByteVectorValues(field);
    for (int i = 0; i < values.size(); i++) {
      final float score =
          fieldInfo.getVectorSimilarityFunction().compare(target, values.vectorValue(i));
      knnCollector.collect(values.ordToDoc(i), score);
      if (knnCollector.earlyTerminated()) {
        return;
      }
    }
  }

  abstract NeighborQueue scorePostingLists(
      FieldInfo fieldInfo,
      KnnCollector knnCollector,
      IVFUtils.CentroidQueryScorer centroidQueryScorer,
      int nProbe)
      throws IOException;

  @Override
  public void close() throws IOException {
    IOUtils.close(rawVectorsReader, ivfCentroids, ivfClusters);
  }

  protected record FieldEntry(
      VectorSimilarityFunction similarityFunction,
      VectorEncoding vectorEncoding,
      long centroidOffset,
      long centroidLength,
      long[] postingListOffsets,
      float[] globalCentroid,
      float globalCentroidDp) {
    IndexInput centroidSlice(IndexInput centroidFile) throws IOException {
      return centroidFile.slice("centroids", centroidOffset, centroidLength);
    }

    //    IndexInput postingsSlice(IndexInput postingsFile, int i) throws IOException {
    //      return postingsFile.slice(
    //          "postings-" + i, postingListOffsetsAndLengths[i][0],
    // postingListOffsetsAndLengths[i][1]);
    //    }
  }

  protected abstract IVFUtils.PostingVisitor getPostingVisitor(
      FieldInfo fieldInfo, IndexInput postingsLists, float[] target, IntPredicate needsScoring)
      throws IOException;

  /**
   * A record containing the centroid and the index offset for a posting list with the given score
   */
  protected record PostingListWithFileOffsetWithScore(
      PostingListWithFileOffset postingListWithFileOffset, float score) {}

  /** A record containing the centroid and the index offset for a posting list */
  // TODO UNIFY THESE TYPES BETWEEN WRITERS & READERS
  public record PostingListWithFileOffset(int centroidOrdinal, long[] fileOffsetAndLength) {}
}
