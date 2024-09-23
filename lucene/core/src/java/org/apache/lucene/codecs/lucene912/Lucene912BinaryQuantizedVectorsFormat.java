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
package org.apache.lucene.codecs.lucene912;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Codec for encoding/decoding binary quantized vectors The binary quantization format used here
 * reflects <a href="https://arxiv.org/abs/2405.12497">RaBitQ</a>. Also see {@link
 * org.apache.lucene.util.quantization.BinaryQuantizer}. Some of key features of RabitQ are:
 *
 * <ul>
 *   <li>Estimating the distance between two vectors using their centroid normalized distance. This
 *       requires some additional corrective factors, but allows for centroid normalization to occur
 *       and thus enabling binary quantization.
 *   <li>Binary quantization of centroid normalized vectors.
 *   <li>Asymmetric quantization of vectors, where query vectors are quantized to half-byte
 *       precision (normalized to the centroid) and then compared directly against the single bit
 *       quantized vectors in the index.
 *   <li>Transforming the half-byte quantized query vectors in such a way that the comparison with
 *       single bit vectors can be done with hamming distance.
 *   <li>Utilizing an error bias calculation enabled by the centroid normalization. This allows for
 *       dynamic rescoring of vectors that fall outside a certain error threshold.
 * </ul>
 *
 * The format is stored in two files:
 *
 * <h2>.veb (vector data) file</h2>
 *
 * <p>Stores the binary quantized vectors in a flat format. Additionally, it stores each vector's
 * corrective factors. At the end of the file, additional information is stored for vector ordinal
 * to centroid ordinal mapping and sparse vector information.
 *
 * <ul>
 *   <li>For each vector:
 *       <ul>
 *         <li><b>[byte]</b> the binary quantized values, each byte holds 8 bits.
 *         <li><b>[float]</b> the corrective values. Two floats for Euclidean distance. Three floats
 *             for the dot-product family of distances.
 *       </ul>
 *   <li>After the vectors, sparse vector information keeping track of monotonic blocks.
 * </ul>
 *
 * <h2>.vemb (vector metadata) file</h2>
 *
 * <p>Stores the metadata for the vectors. This includes the number of vectors, the number of
 * dimensions, centroids and file offset information.
 *
 * <ul>
 *   <li><b>int</b> the field number
 *   <li><b>int</b> the vector encoding ordinal
 *   <li><b>int</b> the vector similarity ordinal
 *   <li><b>vint</b> the vector dimensions
 *   <li><b>vlong</b> the offset to the vector data in the .veb file
 *   <li><b>vlong</b> the length of the vector data in the .veb file
 *   <li><b>vint</b> the number of vectors
 *   <li><b>[float]</b> the centroid of the vectors
 *   <li>The sparse vector information, if required, mapping vector ordinal to doc ID
 * </ul>
 */
public class Lucene912BinaryQuantizedVectorsFormat extends FlatVectorsFormat {

  public static final String BINARIZED_VECTOR_COMPONENT = "BVEC";
  public static final String NAME = "Lucene912BinaryQuantizedVectorsFormat";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;
  static final String META_CODEC_NAME = "Lucene912BinaryQuantizedVectorsFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "Lucene912BinaryQuantizedVectorsFormatData";
  static final String META_EXTENSION = "vemb";
  static final String VECTOR_DATA_EXTENSION = "veb";
  static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

  private static final FlatVectorsFormat rawVectorFormat =
      new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());

  private final BinaryFlatVectorsScorer scorer =
      new Lucene912BinaryFlatVectorsScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());

  /** Creates a new instance with the default number of vectors per cluster. */
  public Lucene912BinaryQuantizedVectorsFormat() {
    super(NAME);
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Lucene912BinaryQuantizedVectorsWriter(
        scorer, rawVectorFormat.fieldsWriter(state), state);
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene912BinaryQuantizedVectorsReader(
        state, rawVectorFormat.fieldsReader(state), scorer);
  }

  @Override
  public String toString() {
    return "Lucene912BinaryQuantizedVectorsFormat(name="
        + NAME
        + ", flatVectorScorer="
        + scorer
        + ")";
  }
}
