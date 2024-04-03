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

import org.apache.lucene.codecs.FlatVectorsFormat;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

public final class Lucene99FlatVectorsFormatOffHeapOptimized extends FlatVectorsFormat {

  private final Lucene99FlatVectorsFormat delegate = new Lucene99FlatVectorsFormat();

  /** Constructs a format */
  public Lucene99FlatVectorsFormatOffHeapOptimized() {
    super();
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Lucene99FlatVectorsWriterOffHeap(delegate.fieldsWriter(state));
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene99FlatVectorsReaderOffHeap(delegate.fieldsReader(state));
  }

  @Override
  public String toString() {
    return "Lucene99FlatVectorsFormat()";
  }
}
