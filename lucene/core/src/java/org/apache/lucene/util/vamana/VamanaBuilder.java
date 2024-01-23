package org.apache.lucene.util.vamana;

import org.apache.lucene.util.InfoStream;

import java.io.IOException;

public interface VamanaBuilder {
  /**
   * Adds all nodes to the graph up to the provided {@code maxOrd}.
   *
   * @param maxOrd The maximum ordinal (excluded) of the nodes to be added.
   */
  OnHeapVamanaGraph build(int maxOrd) throws IOException;

  /** Inserts a doc with vector value to the graph */
  void addGraphNode(int node) throws IOException;

  void finish() throws IOException;

  /** Set info-stream to output debugging information */
  void setInfoStream(InfoStream infoStream);

  OnHeapVamanaGraph getGraph();
}
