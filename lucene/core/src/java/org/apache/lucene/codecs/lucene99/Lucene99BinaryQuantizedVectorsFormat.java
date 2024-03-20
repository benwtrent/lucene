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

import java.io.IOException;
import org.apache.lucene.codecs.FlatVectorsFormat;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Format supporting vector quantization, storage, and retrieval
 *
 * @lucene.experimental
 */
public final class Lucene99BinaryQuantizedVectorsFormat extends FlatVectorsFormat {
  static final String NAME = "Lucene99BinaryQuantizedVectorsFormat";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;
  static final String META_CODEC_NAME = "Lucene99BinaryQuantizedVectorsFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "Lucene99BinaryQuantizedVectorsFormatData";
  static final String META_EXTENSION = "vemb";
  static final String VECTOR_DATA_EXTENSION = "veb";

  private static final FlatVectorsFormat rawVectorFormat =
      new Lucene99ScalarQuantizedVectorsFormat();

  /** Constructs a format using default graph construction parameters */
  public Lucene99BinaryQuantizedVectorsFormat() {}

  @Override
  public String toString() {
    return NAME + "(name=" + NAME + ", rawVectorFormat=" + rawVectorFormat + ")";
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Lucene99BinaryQuantizedVectorsWriter(state, rawVectorFormat.fieldsWriter(state));
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene99BinaryQuantizedVectorsReader(state, rawVectorFormat.fieldsReader(state));
  }
}
