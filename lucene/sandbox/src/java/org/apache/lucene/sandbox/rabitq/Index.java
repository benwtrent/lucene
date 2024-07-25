package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

public class Index {
  // FIXME: introduce logging instead of println
  // FIXME: stream in X so it can be loaded into memory
  // FIXME: better arg parsing
  //  static final int QUORA_E5_DOC_SIZE = 522_931;
  //  static final int SIFTSMALL_DOCS_SIZE = 10_000;

  public static void main(String[] args) throws Exception {
    String source = args[0];
    String dataset = args[1];
    int numCentroids = Integer.parseInt(args[2]);
    int dimensions = Integer.parseInt(args[3]);
    int totalTrainingDocs = Integer.parseInt(args[4]);
    int numDocs = totalTrainingDocs;
    Path basePath = Paths.get(source);
    Path fvecPath = Paths.get(basePath.toString(), dataset + "_base.fvecs");
    int D = dimensions;
    int B = (D + 63) / 64 * 64;
    try (MMapDirectory directory = new MMapDirectory(basePath);
        IndexInput vectorInput = directory.openInput(fvecPath.toString(), IOContext.DEFAULT)) {
      RandomAccessVectorValues.Floats vectorValues =
          new VectorsReaderWithOffset(vectorInput, numDocs, dimensions, Float.BYTES);
      System.out.println("Clustering - " + dataset);
      long startTime = System.nanoTime();
      IVFOutput ivfOutput = clusterWithIVF(vectorValues, numCentroids, dimensions);
      long nanosToComputeIVF = System.nanoTime() - startTime;
      System.out.println(
          "Time to compute IVF: " + TimeUnit.NANOSECONDS.toMillis(nanosToComputeIVF));
      System.out.println("Generating subspaces - " + dataset);
      int MAX_BD = Math.max(D, B);
      startTime = System.nanoTime();
      SubspaceOutput subspaceOutput =
          generateSubSpaces(
              D, B, MAX_BD, vectorValues, ivfOutput.centroidVectors(), ivfOutput.clusterIds());
      System.out.println(
          "Time to compute sub-spaces: "
              + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startTime));
      float[][] centroids = subspaceOutput.cp();
      float[] x0 = subspaceOutput.x0();
      float[] distToCentroid = ivfOutput.distToCentroids();
      int[] clusterId = ivfOutput.clusterIds();
      byte[][] binary = subspaceOutput.repackedBinXP();

      System.out.println("Loading Complete!");
      IVFRN ivf = new IVFRN(numDocs, centroids, distToCentroid, x0, clusterId, binary, dimensions);

      System.out.println("Saving");
      String indexPath = source + "ivfrabitq" + numCentroids + "_B" + B + ".index";
      ivf.save(indexPath);
    }
  }

  private static IVFOutput clusterWithIVF(
      RandomAccessVectorValues.Floats vectors, int numCentroids, int dimensions)
      throws IOException {

    // cluster data vectors
    System.out.println("cluster data vectors");
    SamplingIVF index = new SamplingIVF(numCentroids);
    index.train(vectors, dimensions);
    Centroid[] centroids = index.getCentroids();

    SearchResult[] results = index.search(vectors);
    float[] distToCentroids = new float[results.length];
    int[] clusterIds = new int[results.length];
    for (int i = 0; i < results.length; i++) {
      SearchResult result = results[i];
      float distToCentroid = result.getDistToCentroid();
      clusterIds[i] = result.getClusterId();
      distToCentroids[i] = (float) Math.pow(distToCentroid, 0.5);
    }
    float[][] centroidVectors =
        Arrays.stream(centroids).map(Centroid::getVector).toArray(float[][]::new);
    return new IVFOutput(distToCentroids, clusterIds, centroidVectors);
  }

  private static SubspaceOutput generateSubSpaces(
      int D,
      int B,
      int MAX_BD,
      RandomAccessVectorValues.Floats vectorValues,
      float[][] centroids,
      int[] clusterIds)
      throws IOException {
    int XLength = vectorValues.size();

    float[][] CP = MatrixUtils.padColumns(centroids, MAX_BD - D); // typically no-op if D/64

    float[] x0 = new float[XLength];
    byte[][] repackedBinXP = new byte[XLength][B / 8];

    for (int i = 0; i < vectorValues.size(); i++) {
      float[] X = vectorValues.vectorValue(i);
      float[] XP = MatrixUtils.partialPadColumns(X, MAX_BD - D); // typically no-op if D/64

      MatrixUtils.partialSubtract(XP, CP[clusterIds[i]]);

      // The inner product between the data vector and the quantized data vector
      float norm = MatrixUtils.partialNormForRow(XP);
      float[] XPSubset = MatrixUtils.partialSubset(XP, B); // typically no-op if D/64
      MatrixUtils.partialRemoveSignAndDivide(XPSubset, (float) Math.pow(B, 0.5));
      x0[i] = MatrixUtils.partialSumAndNormalize(XPSubset, norm);
      repackedBinXP[i] = MatrixUtils.partialRepackAsUInt64(XP, B);
    }
    return new SubspaceOutput(CP, x0, repackedBinXP);
  }
}
