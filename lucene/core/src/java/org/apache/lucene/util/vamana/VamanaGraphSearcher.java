package org.apache.lucene.util.vamana;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class VamanaGraphSearcher {

  public record CachedNode(float[] vector, int[] neighbors) {}

  /**
   * Scratch data structures that are used in each {@link #search} call. These can be expensive to
   * allocate, so they're cleared and reused across calls.
   */
  private final NeighborQueue candidates;

  private final Map<Integer, CachedNode> cache;

  private BitSet visited;

  /**
   * Creates a new graph searcher.
   *
   * @param candidates max heap that will track the candidate nodes to explore
   * @param visited bit set that will track nodes that have already been visited
   */
  public VamanaGraphSearcher(NeighborQueue candidates, BitSet visited) {
    this(candidates, visited, null);
  }

  public VamanaGraphSearcher(
    NeighborQueue candidates, BitSet visited, Map<Integer, CachedNode> cache) {
    this.candidates = candidates;
    this.visited = visited;
    this.cache = cache;
  }

  /**
   * Searches HNSW graph for the nearest neighbors of a query vector.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param knnCollector a collector of top knn results to be returned
   * @param graph the graph values. May represent the entire graph, or a level in a hierarchical
   *     graph.
   * @param acceptOrds {@link Bits} that represents the allowed document ordinals to match, or
   *     {@code null} if they are all allowed to match.
   */
  public static void search(
    RandomVectorScorer scorer,
    KnnCollector knnCollector,
    VamanaGraph graph,
    Bits acceptOrds,
    Map<Integer, CachedNode> cache)
    throws IOException {
    VamanaGraphSearcher graphSearcher =
      new VamanaGraphSearcher(
        new NeighborQueue(knnCollector.k(), true),
        new SparseFixedBitSet(getGraphSize(graph)),
        cache);
    search(scorer, knnCollector, graph, graphSearcher, acceptOrds);
  }

  /**
   * Search {@link OnHeapVamanaGraph}, this method is thread safe.
   *
   * @param scorer the scorer to compare the query with the nodes
   * @param topK the number of nodes to be returned
   * @param graph the graph values. May represent the entire graph, or a level in a hierarchical
   *     graph.
   * @param acceptOrds {@link Bits} that represents the allowed document ordinals to match, or
   *     {@code null} if they are all allowed to match.
   * @param visitedLimit the maximum number of nodes that the search is allowed to visit
   * @return a set of collected vectors holding the nearest neighbors found
   */
  public static KnnCollector search(
    RandomVectorScorer scorer,
    int topK,
    OnHeapVamanaGraph graph,
    Bits acceptOrds,
    int visitedLimit)
    throws IOException {
    KnnCollector knnCollector = new TopKnnCollector(topK, visitedLimit);
    OnHeapVamanaGraphSearcher graphSearcher =
      new OnHeapVamanaGraphSearcher(
        new NeighborQueue(topK, true), new SparseFixedBitSet(getGraphSize(graph)));
    search(scorer, knnCollector, graph, graphSearcher, acceptOrds);
    return knnCollector;
  }

  private static void search(
    RandomVectorScorer scorer,
    KnnCollector knnCollector,
    VamanaGraph graph,
    VamanaGraphSearcher graphSearcher,
    Bits acceptOrds)
    throws IOException {
    int initialEp = graph.entryNode();
    if (initialEp == -1) {
      return;
    }

    graphSearcher.search(knnCollector, scorer, new int[] {initialEp}, graph, acceptOrds);
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in REVERSE
   * proximity order -- the most distant neighbor of the topK found, i.e. the one with the lowest
   * score/comparison value, will be at the top of the heap, while the closest neighbor will be the
   * last to be popped.
   */
  void search(
    KnnCollector results,
    RandomVectorScorer scorer,
    final int[] eps,
    VamanaGraph graph,
    Bits acceptOrds)
    throws IOException {

    int size = getGraphSize(graph);

    prepareScratchState(size);

    for (int ep : eps) {
      if (visited.getAndSet(ep) == false) {
        if (results.earlyTerminated()) {
          break;
        }
        float score = scorer.score(ep);
        results.incVisitedCount(1);
        candidates.add(ep, score);
        if (acceptOrds == null || acceptOrds.get(ep)) {
          results.collect(ep, score);
        }
      }
    }
    sequentialSearch(results, scorer, graph, acceptOrds, size);
  }

  private void sequentialSearch(
    KnnCollector results, RandomVectorScorer scorer, VamanaGraph graph, Bits acceptOrds, int size)
    throws IOException {
    float minAcceptedSimilarity = results.minCompetitiveSimilarity();
    while (candidates.size() > 0 && results.earlyTerminated() == false) {
      // get the best candidate (closest or best scoring)
      float topCandidateSimilarity = candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        break;
      }

      int topCandidateNode = candidates.pop();
      int friendOrd;
      VamanaGraph.NodesIterator neighbors = getNeighbors(results, graph, topCandidateNode);
      while (neighbors.hasNext()) {
        friendOrd = neighbors.nextInt();
        assert friendOrd < size : "friendOrd=" + friendOrd + "; size=" + size;
        if (visited.getAndSet(friendOrd)) {
          continue;
        }

        if (results.earlyTerminated()) {
          break;
        }
        float friendSimilarity = scorer.score(friendOrd);
        results.incVisitedCount(1);
        if (friendSimilarity >= minAcceptedSimilarity) {
          candidates.add(friendOrd, friendSimilarity);
          if (acceptOrds == null || acceptOrds.get(friendOrd)) {
            if (results.collect(friendOrd, friendSimilarity)) {
              minAcceptedSimilarity = results.minCompetitiveSimilarity();
            }
          }
        }
      }
    }
  }

  private void prepareScratchState(int capacity) {
    candidates.clear();
    if (visited.length() < capacity) {
      visited = FixedBitSet.ensureCapacity((FixedBitSet) visited, capacity);
    }
    visited.clear();
  }

  /**
   * Seek a specific node in the given graph. The default implementation will just call {@link
   * VamanaGraph#seek(int)}
   *
   * @throws IOException when seeking the graph
   */
  void graphSeek(VamanaGraph graph, int targetNode) throws IOException {
    graph.seek(targetNode);
  }

  VamanaGraph.NodesIterator getNeighbors(KnnCollector results, VamanaGraph graph, int targetNode)
    throws IOException {
    if (cache == null || !cache.containsKey(targetNode)) {
      // Not using the cache, so need to seek in the graph (IO happens here).
      graph.seek(targetNode);
      return graph.getNeighbors();
    }

    var cached = cache.get(targetNode);
    return new VamanaGraph.ArrayNodesIterator(cached.neighbors, cached.neighbors.length);
  }

  /**
   * Get the next neighbor from the graph, you must call {@link #graphSeek(VamanaGraph, int)} before
   * calling this method. The default implementation will just call {@link
   * VamanaGraph#nextNeighbor()}
   *
   * @return see {@link VamanaGraph#nextNeighbor()}
   * @throws IOException when advance neighbors
   */
  int graphNextNeighbor(VamanaGraph graph) throws IOException {
    return graph.nextNeighbor();
  }

  private static int getGraphSize(VamanaGraph graph) {
    return graph.maxNodeId() + 1;
  }

  /**
   * This class allows {@link OnHeapVamanaGraph} to be searched in a thread-safe manner by avoiding
   * the unsafe methods (seek and nextNeighbor, which maintain state in the graph object) and
   * instead maintaining the state in the searcher object.
   *
   * <p>Note the class itself is NOT thread safe, but since each search will create a new Searcher,
   * the search methods using this class are thread safe.
   */
  private static class OnHeapVamanaGraphSearcher extends VamanaGraphSearcher {

    private NeighborArray cur;
    private int upto;

    private OnHeapVamanaGraphSearcher(NeighborQueue candidates, BitSet visited) {
      super(candidates, visited);
    }

    @Override
    void graphSeek(VamanaGraph graph, int targetNode) {
      cur = ((OnHeapVamanaGraph) graph).getNeighbors(targetNode);
      upto = -1;
    }

    @Override
    int graphNextNeighbor(VamanaGraph graph) {
      if (++upto < cur.size()) {
        return cur.nodes[upto];
      }
      return NO_MORE_DOCS;
    }
  }

}
