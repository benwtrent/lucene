/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.codecs.quantization;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.GroupVIntUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.VectorUtil;

/**
 * @lucene.experimental
 */
public abstract class IVFVectorsWriter extends KnnVectorsWriter {

  private final List<FieldWriter> fieldWriters = new ArrayList<>();
  private final IndexOutput ivfCentroids, ivfClusters;
  private final IndexOutput ivfMeta;
  private final FlatVectorsWriter rawVectorDelegate;
  private final SegmentWriteState segmentWriteState;

  protected IVFVectorsWriter(SegmentWriteState state, FlatVectorsWriter rawVectorDelegate)
      throws IOException {
    this.segmentWriteState = state;
    this.rawVectorDelegate = rawVectorDelegate;
    final String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.IVF_META_EXTENSION);

    final String ivfCentroidsFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.CENTROID_EXTENSION);
    final String ivfClustersFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.CLUSTER_EXTENSION);
    boolean success = false;
    try {
      ivfMeta = state.directory.createOutput(metaFileName, state.context);
      CodecUtil.writeIndexHeader(
          ivfMeta,
          IVFVectorsFormat.NAME,
          IVFVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      ivfCentroids = state.directory.createOutput(ivfCentroidsFileName, state.context);
      CodecUtil.writeIndexHeader(
          ivfCentroids,
          IVFVectorsFormat.NAME,
          IVFVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      ivfClusters = state.directory.createOutput(ivfClustersFileName, state.context);
      CodecUtil.writeIndexHeader(
          ivfClusters,
          IVFVectorsFormat.NAME,
          IVFVectorsFormat.VERSION_CURRENT,
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
  public final KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    if (fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE) {
      throw new IllegalArgumentException("IVF does not support cosine similarity");
    }
    final FlatFieldVectorsWriter<?> rawVectorDelegate = this.rawVectorDelegate.addField(fieldInfo);
    if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
      @SuppressWarnings("unchecked")
      final FlatFieldVectorsWriter<float[]> floatWriter =
          (FlatFieldVectorsWriter<float[]>) rawVectorDelegate;
      fieldWriters.add(new FieldWriter(fieldInfo, floatWriter));
    }
    return rawVectorDelegate;
  }

  protected abstract SortedAssignments calculateAndWriteCentroids(
      FieldInfo fieldInfo,
      FloatVectorValues floatVectorValues,
      IndexOutput temporaryCentroidOutput,
      MergeState mergeState,
      float[] globalCentroid)
      throws IOException;

  protected abstract long[] buildAndWritePostingsLists(
      FieldInfo fieldInfo,
      IVFUtils.CentroidAssignmentScorer scorer,
      FloatVectorValues floatVectorValues,
      IndexOutput postingsOutput,
      MergeState mergeState,
      Set<SortedAssignment> sortedAssignments)
      throws IOException;

  protected abstract IVFUtils.CentroidAssignmentScorer calculateAndWriteCentroids(
      FieldInfo fieldInfo,
      FloatVectorValues floatVectorValues,
      IndexOutput centroidOutput,
      float[] globalCentroid)
      throws IOException;

  protected abstract long[] buildAndWritePostingsLists(
      FieldInfo fieldInfo,
      InfoStream infoStream,
      IVFUtils.CentroidAssignmentScorer scorer,
      FloatVectorValues floatVectorValues,
      IndexOutput postingsOutput)
      throws IOException;

  protected abstract IVFUtils.CentroidAssignmentScorer createCentroidScorer(
      IndexInput centroidsInput, int numCentroids, FieldInfo fieldInfo, float[] globalCentroid)
      throws IOException;

  @Override
  public final void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    rawVectorDelegate.flush(maxDoc, sortMap);
    for (FieldWriter fieldWriter : fieldWriters) {
      float[] globalCentroid = new float[fieldWriter.fieldInfo.getVectorDimension()];
      VectorUtil.calculateCentroid(fieldWriter.delegate().getVectors(), globalCentroid);
      // build a float vector values with random access
      // built projections
      final FloatVectorValues raw = FloatVectorValues.fromFloats(fieldWriter.delegate.getVectors(), fieldWriter.fieldInfo.getVectorDimension());
      IVFUtils.RandomOrtho ortho = IVFUtils.forDims(fieldWriter.fieldInfo.getVectorDimension());
      int[][] assignments = IVFUtils.varianceBlockAssignments(raw, ortho);
      final FloatVectorValues floatVectorValues =
        getFloatVectorValues(fieldWriter.fieldInfo, fieldWriter.delegate, assignments, ortho, maxDoc);
      // build centroids
      long centroidOffset = ivfCentroids.alignFilePointer(Float.BYTES);
      final IVFUtils.CentroidAssignmentScorer centroidAssignmentScorer =
          calculateAndWriteCentroids(
              fieldWriter.fieldInfo, floatVectorValues, ivfCentroids, globalCentroid);
      long centroidLength = ivfCentroids.getFilePointer() - centroidOffset;
      final long[] offsets =
          buildAndWritePostingsLists(
              fieldWriter.fieldInfo,
              segmentWriteState.infoStream,
              centroidAssignmentScorer,
              floatVectorValues,
              ivfClusters);
      // write posting lists
      writeMeta(fieldWriter.fieldInfo, centroidOffset, centroidLength, offsets, globalCentroid, assignments, ortho);
    }
  }

  private static FloatVectorValues getFloatVectorValues(
      FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> fieldVectorsWriter, int[][] assignments, IVFUtils.RandomOrtho ortho,  int maxDoc)
      throws IOException {
    // we can copy the vectors in place as the base writer has already flushed
    List<float[]> rawVectors = fieldVectorsWriter.getVectors();
    List<float[]> vectors = new ArrayList<>(rawVectors.size());
    for (int i = 0; i < rawVectors.size(); i++) {
      float[] vector = rawVectors.get(i);
      IVFUtils.randomOrthogonalTransform(rawVectors.get(i), ortho, assignments, vector);
      vectors.add(vector);
    }
    if (vectors.size() == maxDoc) {
      return FloatVectorValues.fromFloats(vectors, fieldInfo.getVectorDimension());
    }
    final DocIdSetIterator iterator = fieldVectorsWriter.getDocsWithFieldSet().iterator();
    final int[] docIds = new int[vectors.size()];
    for (int i = 0; i < docIds.length; i++) {
      docIds[i] = iterator.nextDoc();
    }
    assert iterator.nextDoc() == NO_MORE_DOCS;
    return new FloatVectorValues() {
      @Override
      public float[] vectorValue(int ord) {
        return vectors.get(ord);
      }

      @Override
      public FloatVectorValues copy() {
        return this;
      }

      @Override
      public int dimension() {
        return fieldInfo.getVectorDimension();
      }

      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int ordToDoc(int ord) {
        return docIds[ord];
      }
    };
  }

  static IVFVectorsReader getIVFReader(KnnVectorsReader vectorsReader, String fieldName) {
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
      vectorsReader = candidateReader.getFieldReader(fieldName);
    }
    if (vectorsReader instanceof IVFVectorsReader reader) {
      return reader;
    }
    return null;
  }

  record SortedAssignments(int numCentroids, Set<SortedAssignment> assignments) {}

  public record SortedAssignment(int docId, short centroid, float distance, boolean isSoar)
      implements Comparable<SortedAssignment> {
    @Override
    public int compareTo(SortedAssignment o) {
      return Float.compare(this.distance, o.distance);
    }
  }

  @Override
  public final void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    rawVectorDelegate.mergeOneField(fieldInfo, mergeState);
    if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
      final int numVectors;
      String name = null;
      boolean success = false;
      // build a float vector values with random access. In order to do that we dump the vectors to
      // a temporary file
      // and write the docID follow by the vector
      FloatVectorValues raw = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
      IVFUtils.RandomOrtho ortho = IVFUtils.forDims(fieldInfo.getVectorDimension());
      int[][] projectedAssignments = IVFUtils.varianceBlockAssignments(raw, ortho);
      try (IndexOutput out =
          mergeState.segmentInfo.dir.createTempOutput(
              mergeState.segmentInfo.name, "ivf_", IOContext.DEFAULT)) {
        name = out.getName();
        numVectors =
            writeFloatVectorValues(
                fieldInfo, out, ortho, projectedAssignments, MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState));
        success = true;
      } finally {
        if (success == false && name != null) {
          IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, name);
        }
      }
      float[] globalCentroid = new float[fieldInfo.getVectorDimension()];
      int vectorCount = 0;
      for (var knnReaders : mergeState.knnVectorsReaders) {
        IVFVectorsReader ivfReader = getIVFReader(knnReaders, fieldInfo.name);
        if (ivfReader != null) {
          int numVecs = ivfReader.getFloatVectorValues(fieldInfo.name).size();
          float[] readerGlobalCentroid = ivfReader.getGlobalCentroid(fieldInfo);
          if (readerGlobalCentroid != null) {
            vectorCount += numVecs;
            for (int i = 0; i < globalCentroid.length; i++) {
              globalCentroid[i] += readerGlobalCentroid[i] * numVecs;
            }
          }
        }
      }
      if (vectorCount > 0) {
        for (int i = 0; i < globalCentroid.length; i++) {
          globalCentroid[i] /= vectorCount;
        }
      }
      try (IndexInput in = mergeState.segmentInfo.dir.openInput(name, IOContext.DEFAULT)) {
        final FloatVectorValues floatVectorValues = getFloatVectorValues(fieldInfo, in, numVectors);
        success = false;
        IVFUtils.CentroidAssignmentScorer centroidAssignmentScorer;
        long centroidOffset;
        long centroidLength;
        String centroidTempName = null;
        int numCentroids;
        IndexOutput centroidTemp = null;
        Set<SortedAssignment> sortedAssignments;
        try {
          centroidTemp =
              mergeState.segmentInfo.dir.createTempOutput(
                  mergeState.segmentInfo.name, "civf_", IOContext.DEFAULT);
          centroidTempName = centroidTemp.getName();

          // FIXME: need to merge first before wiring this all together
          SortedAssignments assignments =
              calculateAndWriteCentroids(
                  fieldInfo, floatVectorValues, centroidTemp, mergeState, globalCentroid);
          numCentroids = assignments.numCentroids;
          sortedAssignments = assignments.assignments;

          success = true;
        } finally {
          if (success == false && centroidTempName != null) {
            IOUtils.closeWhileHandlingException(centroidTemp);
            IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, centroidTempName);
          }
        }
        try {
          if (numCentroids == 0) {
            centroidOffset = ivfCentroids.getFilePointer();
            writeMeta(fieldInfo, centroidOffset, 0, new long[0], null, null, null);
            CodecUtil.writeFooter(centroidTemp);
            IOUtils.close(centroidTemp);
            return;
          }
          CodecUtil.writeFooter(centroidTemp);
          IOUtils.close(centroidTemp);
          centroidOffset = ivfCentroids.alignFilePointer(Float.BYTES);
          try (IndexInput centroidInput =
              mergeState.segmentInfo.dir.openInput(centroidTempName, IOContext.DEFAULT)) {
            ivfCentroids.copyBytes(
                centroidInput, centroidInput.length() - CodecUtil.footerLength());
            centroidLength = ivfCentroids.getFilePointer() - centroidOffset;
            centroidAssignmentScorer =
                createCentroidScorer(centroidInput, numCentroids, fieldInfo, globalCentroid);
            assert centroidAssignmentScorer.size() == numCentroids;
            // build a float vector values with random access
            // build centroids
            final long[] offsets =
                buildAndWritePostingsLists(
                    fieldInfo,
                    centroidAssignmentScorer,
                    floatVectorValues,
                    ivfClusters,
                    mergeState,
                    sortedAssignments);
            // write posting lists

            // TODO handle this correctly by creating new centroid
            if (vectorCount == 0 && offsets.length > 0) {
              throw new IllegalStateException(
                  "No global centroid found for field: " + fieldInfo.name);
            }
            assert offsets.length == centroidAssignmentScorer.size();
            writeMeta(fieldInfo, centroidOffset, centroidLength, offsets, globalCentroid, projectedAssignments, ortho);
          }
        } finally {
          IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, name);
          IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, centroidTempName);
        }
      } finally {
        IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, name);
      }
    }
  }

  private static FloatVectorValues getFloatVectorValues(
      FieldInfo fieldInfo, IndexInput randomAccessInput, int numVectors) {
    final long length = (long) Float.BYTES * fieldInfo.getVectorDimension() + Integer.BYTES;
    final float[] vector = new float[fieldInfo.getVectorDimension()];
    return new FloatVectorValues() {
      @Override
      public float[] vectorValue(int ord) throws IOException {
        randomAccessInput.seek(ord * length + Integer.BYTES);
        randomAccessInput.readFloats(vector, 0, vector.length);
        return vector;
      }

      @Override
      public FloatVectorValues copy() {
        return this;
      }

      @Override
      public int dimension() {
        return fieldInfo.getVectorDimension();
      }

      @Override
      public int size() {
        return numVectors;
      }

      @Override
      public int ordToDoc(int ord) {
        try {
          randomAccessInput.seek(ord * length);
          return randomAccessInput.readInt();
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }
    };
  }

  private static int writeFloatVectorValues(
    FieldInfo fieldInfo, IndexOutput out, IVFUtils.RandomOrtho ortho, int[][] assignments, FloatVectorValues floatVectorValues)
      throws IOException {
    int numVectors = 0;
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    final KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
    for (int docV = iterator.nextDoc(); docV != NO_MORE_DOCS; docV = iterator.nextDoc()) {
      numVectors++;
      float[] vector = floatVectorValues.vectorValue(iterator.index());
      if (assignments != null) {
        IVFUtils.randomOrthogonalTransform(vector, ortho, assignments, vector);
      }
      out.writeInt(iterator.docID());
      buffer.asFloatBuffer().put(vector);
      out.writeBytes(buffer.array(), buffer.array().length);
    }
    return numVectors;
  }

  private void writeMeta(
      FieldInfo field,
      long centroidOffset,
      long centroidLength,
      long[] offsets,
      float[] globalCentroid,
      int[][] assignments,
      IVFUtils.RandomOrtho ortho
  )
      throws IOException {
    ivfMeta.writeInt(field.number);
    ivfMeta.writeInt(field.getVectorEncoding().ordinal());
    ivfMeta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    ivfMeta.writeLong(centroidOffset);
    ivfMeta.writeLong(centroidLength);
    ivfMeta.writeVInt(offsets.length);
    for (long offset : offsets) {
      ivfMeta.writeLong(offset);
    }
    if (offsets.length > 0) {
      final ByteBuffer buffer =
        ByteBuffer.allocate(globalCentroid.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      buffer.asFloatBuffer().put(globalCentroid);
      ivfMeta.writeBytes(buffer.array(), buffer.array().length);
      ivfMeta.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(globalCentroid, globalCentroid)));
      ivfMeta.writeVInt(assignments.length);
      for (int[] assignment : assignments) {
        ivfMeta.writeVInt(assignment.length);
        for (int i = 0; i < assignment.length; i++) {
          ivfMeta.writeVInt(assignment[i]);
        }
      }
      ivfMeta.writeVInt(ortho.matrix().length);
      for (int i = 0; i < ortho.dimBlocks().length; i++) {
        ivfMeta.writeVInt(ortho.dimBlocks()[i]);
      }
      for (int i = 0; i < ortho.matrix().length; i++) {
        float[] floats = ortho.matrix()[i];
        final ByteBuffer matrixBuffer =
          ByteBuffer.allocate(floats.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        matrixBuffer.asFloatBuffer().put(floats);
        ivfMeta.writeBytes(matrixBuffer.array(), matrixBuffer.array().length);
      }
    }
  }

  private static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < SIMILARITY_FUNCTIONS.size(); i++) {
      if (SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  @Override
  public final void finish() throws IOException {
    rawVectorDelegate.finish();
    if (ivfMeta != null) {
      // write end of fields marker
      ivfMeta.writeInt(-1);
      CodecUtil.writeFooter(ivfMeta);
    }
    if (ivfCentroids != null) {
      CodecUtil.writeFooter(ivfCentroids);
    }
    if (ivfClusters != null) {
      CodecUtil.writeFooter(ivfClusters);
    }
  }

  @Override
  public final void close() throws IOException {
    IOUtils.close(rawVectorDelegate, ivfMeta, ivfCentroids, ivfClusters);
  }

  @Override
  public final long ramBytesUsed() {
    return rawVectorDelegate.ramBytesUsed();
  }

  private record FieldWriter(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> delegate) {}
}
