package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.PrintStreamInfoStream;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

public class Search {

  //  public static final int TOTAL_QUERY_VECTORS_QUORA = 1000;
  //  public static final int TOTAL_QUERY_VECTORS_SIFTSMALL = 100;
  static VectorSimilarityFunction vectorFunction = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

  public static void main(String[] args) throws Exception {
    // FIXME: better arg parsing
    // FIXME: clean up gross path mgmt
    // search DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS B_QUERY OUTPUT_PATH
    String source = args[0]; // eg "/Users/jwagster/Desktop/gist1m/gist/"
    String dataset = args[1]; // eg "gist"
    int numCentroids = Integer.parseInt(args[2]);
    int dimensions = Integer.parseInt(args[3]);
    int k = Integer.parseInt(args[4]);
    int totalQueryVectors = Integer.parseInt(args[5]);
    boolean doHnsw = false;
    int maxConns = 16;
    int beamWidth = 100;
    if (args.length > 6) {
      doHnsw = Boolean.parseBoolean(args[6]);
      if (args.length > 8) {
        maxConns = Integer.parseInt(args[7]);
        beamWidth = Integer.parseInt(args[8]);
      }
    }
    InfoStream infoStream = new PrintStreamInfoStream(System.out);
    int B = (dimensions + 63) / 64 * 64;
    Path basePath = Paths.get(source);

    String queryPath = String.format("%s%s_query.fvecs", source, dataset);
    String dataPath = String.format("%s%s_base.fvecs", source, dataset);
    String groundTruthPath = String.format("%s%s_mip_groundtruth.fvecs", source, dataset);
    String indexPath = String.format("%sivfrabitq%d_B%d.index", source, numCentroids, B);
    IVFRN ivfrn = IVFRN.load(indexPath);
    System.out.println("Loaded IVFRN index with size: " + ivfrn.getN());
    try (MMapDirectory directory = new MMapDirectory(basePath);
        IndexInput vectorInput = directory.openInput(dataPath, IOContext.DEFAULT);
        IndexInput queryInput = directory.openInput(queryPath, IOContext.READONCE)) {
      RandomAccessVectorValues.Floats queryVectors =
          new VectorsReaderWithOffset(queryInput, totalQueryVectors, dimensions, Float.BYTES);
      RandomAccessVectorValues.Floats dataVectors =
          new VectorsReaderWithOffset(vectorInput, ivfrn.getN(), dimensions, Float.BYTES);
      int[][] G = new int[totalQueryVectors][k];
      if (Files.exists(directory.getDirectory().resolve(groundTruthPath))) {
        G = IOUtils.readGroundTruth(groundTruthPath, directory, totalQueryVectors);
      } else {
        // writing to the ground truth file
        System.out.println("Calculating nearest neighbors");
        try (IndexOutput queryGroundTruthOutput =
            directory.createOutput(groundTruthPath, IOContext.DEFAULT)) {
          for (int i = 0; i < totalQueryVectors; i++) {
            float[] candidate = queryVectors.vectorValue(i);
            G[i] = getNN(dataVectors, candidate, k, vectorFunction);
            queryGroundTruthOutput.writeInt(G[i].length);
            for (int doc : G[i]) {
              queryGroundTruthOutput.writeInt(doc);
            }
            if (i % 10 == 0) {
              System.out.print(".");
            }
          }
        }
        System.out.println("Done calculating nearest neighbors");
      }
      if (doHnsw) {
        System.out.println(
            "Building HNSW graph with maxConns=" + maxConns + " beamWidth=" + beamWidth);
        RBQRandomVectorScorerSupplier scorerSupplier =
            new RBQRandomVectorScorerSupplier(dataVectors, ivfrn);
        HnswGraphBuilder hnsw =
            HnswGraphBuilder.create(scorerSupplier, maxConns, beamWidth, 42, dataVectors.size());
        hnsw.setInfoStream(infoStream);
        long graphBuildTime = System.nanoTime();
        OnHeapHnswGraph graph = hnsw.build(dataVectors.size());
        graphBuildTime = System.nanoTime() - graphBuildTime;
        System.out.println(
            "Graph build time: " + TimeUnit.NANOSECONDS.toMillis(graphBuildTime) + " ms");
        testHnsw(queryVectors, dataVectors, G, ivfrn, graph, k, k * 5);
      } else {
        test(queryVectors, dataVectors, G, ivfrn, k);
      }
    }
  }

  private static int[] getNN(
      RandomAccessVectorValues.Floats reader,
      float[] query,
      int topK,
      VectorSimilarityFunction vectorFunction)
      throws IOException {
    int[] result = new int[topK];
    NeighborQueue queue = new NeighborQueue(topK, false);
    for (int j = 0; j < reader.size(); j++) {
      float[] doc = reader.vectorValue(j);
      float dist = vectorFunction.compare(query, doc);
      queue.insertWithOverflow(j, dist);
    }
    for (int k = topK - 1; k >= 0; k--) {
      result[k] = queue.topNode();
      queue.pop();
    }
    return result;
  }

  public static void testHnsw(
      RandomAccessVectorValues.Floats queryVectors,
      RandomAccessVectorValues.Floats dataVectors,
      int[][] G,
      IVFRN ivf,
      OnHeapHnswGraph hnsw,
      int k,
      int numCandidates)
      throws IOException {

    float totalUsertime = 0;
    float totalRatio = 0;
    int correctCount = 0;
    long totalVectorComparisons = 0;

    System.out.println("Starting search");
    for (int i = 0; i < queryVectors.size(); i++) {
      long startTime = System.nanoTime();
      float[] queryVector = queryVectors.vectorValue(i);
      RandomVectorScorer scorer =
          new RBQRandomVectorScorerSupplier.RBQRandomVectorScorer(
              queryVector, ivf.quantizeQuery(queryVectors.vectorValue(i)), dataVectors, ivf);
      KnnCollector knnCollector =
          HnswGraphSearcher.search(scorer, numCandidates, hnsw, null, hnsw.size());
      totalVectorComparisons += knnCollector.visitedCount();
      TopDocs collectedDocs = knnCollector.topDocs();
      HitQueue KNNs = new HitQueue(k, false);
      // rescore & get top k
      if (numCandidates > k) {
        for (int j = 0; j < collectedDocs.scoreDocs.length; j++) {
          float rawScore =
              vectorFunction.compare(
                  dataVectors.vectorValue(collectedDocs.scoreDocs[j].doc), queryVector);
          KNNs.insertWithOverflow(new ScoreDoc(collectedDocs.scoreDocs[j].doc, rawScore));
        }
      } else {
        KNNs.addAll(Arrays.asList(knnCollector.topDocs().scoreDocs));
      }
      float usertime =
          (System.nanoTime() - startTime)
              / 1e3f; // convert to microseconds to compare to the c impl
      totalUsertime += usertime;

      int correct = 0;
      while (KNNs.size() > 0) {
        int id = KNNs.pop().doc;
        for (int j = 0; j < k; j++) {
          if (id == G[i][j]) {
            correct++;
          }
        }
      }
      correctCount += correct;

      if (i % 1500 == 0) {
        System.out.print(".");
      }
    }
    System.out.println();

    // FIXME: missing rotation time?
    float timeUsPerQuery = totalUsertime / queryVectors.size();
    float recall = (float) correctCount / (queryVectors.size() * k);
    float averageRatio = totalRatio / (queryVectors.size() * k);
    float avgVectorComparisons = (float) totalVectorComparisons / queryVectors.size();

    System.out.println("------------------------------------------------");
    System.out.println("Recall = " + recall * 100f + "%\t" + "Ratio = " + averageRatio);
    System.out.println(
        "Avg Time Per Search = "
            + timeUsPerQuery
            + " us\t QPS = "
            + (1e6 / timeUsPerQuery)
            + " query/s");
    System.out.println("Total Search Time = " + (totalUsertime / 1e6f) + " sec");
    System.out.println("Avg Vector Comparisons = " + avgVectorComparisons);
  }

  public static void test(
      RandomAccessVectorValues.Floats queryVectors,
      RandomAccessVectorValues.Floats dataVectors,
      int[][] G,
      IVFRN ivf,
      int k)
      throws IOException {

    int nprobes = 300;
    nprobes = Math.min(nprobes, ivf.getC()); // FIXME: hardcoded
    assert nprobes <= k;

    float totalUsertime = 0;
    float totalRatio = 0;
    int correctCount = 0;

    float errorBoundAvg = 0f;
    int maxEstimatorSize = 0;
    int totalEstimatorAdds = 0;
    int floatingPointOps = 0;
    System.out.println("Starting search");
    for (int i = 0; i < queryVectors.size(); i++) {
      long startTime = System.nanoTime();
      float[] queryVector = queryVectors.vectorValue(i);
      IVFRNResult result = ivf.search(dataVectors, queryVector, k, nprobes, vectorFunction);
      PriorityQueue<Result> KNNs = result.results();
      IVFRNStats stats = result.stats();
      float usertime =
          (System.nanoTime() - startTime)
              / 1e3f; // convert to microseconds to compare to the c impl
      totalUsertime += usertime;

      PriorityQueue<Result> copyOfKNN = new PriorityQueue<>();
      copyOfKNN.addAll(KNNs);

      int correct = 0;
      while (!KNNs.isEmpty()) {
        int id = KNNs.remove().c();
        for (int j = 0; j < k; j++) {
          if (id == G[i][j]) {
            correct++;
          }
        }
      }
      correctCount += correct;

      if (i % 15 == 0) {
        System.out.print(".");
      }

      errorBoundAvg += stats.errorBoundAvg();
      maxEstimatorSize = stats.maxEstimatorSize();
      totalEstimatorAdds += stats.totalEstimatorQueueAdds();
      floatingPointOps += stats.floatingPointOps();
    }
    System.out.println();

    // FIXME: missing rotation time?
    float timeUsPerQuery = totalUsertime / queryVectors.size();
    float recall = (float) correctCount / (queryVectors.size() * k);
    float averageRatio = totalRatio / (queryVectors.size() * k);

    System.out.println("------------------------------------------------");
    System.out.println("nprobe = " + nprobes + "\tk = " + k + "\tCoarse Clusters = " + ivf.getC());
    System.out.println("Recall = " + recall * 100f + "%\t" + "Ratio = " + averageRatio);
    System.out.println(
        "Avg Time Per Search = "
            + timeUsPerQuery
            + " us\t QPS = "
            + (1e6 / timeUsPerQuery)
            + " query/s");
    System.out.println("Total Search Time = " + (totalUsertime / 1e6f) + " sec");
    System.out.println("Error Bound Avg = " + (errorBoundAvg / queryVectors.size()));
    System.out.println("Max Estimator Size = " + maxEstimatorSize);
    System.out.println("Total Estimator Adds = " + totalEstimatorAdds);
    System.out.println("Avg Floating Point Ops = " + (floatingPointOps / queryVectors.size()));
  }
}
