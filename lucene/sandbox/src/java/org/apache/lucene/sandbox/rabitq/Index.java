package org.apache.lucene.sandbox.rabitq;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;


public class Index {
    public static void main(String[] args) throws Exception {
        // FIXME: FUTURE - allow the output path to be settable
        // index DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS
        String source ="/Users/benjamintrent/rabit_data/";
        String dataset = "quora-522k-e5small_corpus-quora-E5-small";
        int numCentroids = 1;// Integer.parseInt(args[2]);
        int dimensions = 384;// Integer.parseInt(args[3]);

        // FIXME: FUTURE - switch these to log statements
        System.out.println("Clustering - " + dataset);
        long startTime = System.nanoTime();
        clusterWithIVF(source, dataset, numCentroids);
        System.out.println("Time to compute IVF: " + (System.nanoTime() - startTime) / 1e9);

        // FIXME: FUTURE - switch these to log statements
        System.out.println("Generating subspaces - " + dataset);
        generateSubSpaces(source, dataset, numCentroids);

        int B = (dimensions + 63) / 64 * 64;

        // FIXME: FUTURE - clean up this gross path mgmt
        String dataPath = String.format("%s%s.fvec", source, dataset);
        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));

        String centroidPath = String.format("%sRandCentroid_C%d_B%d.fvecs", source, numCentroids, B);
        float[][] centroids = IOUtils.readFvecs(new FileInputStream(centroidPath));

        String x0Path = String.format("%sx0_C%d_B%d.fvecs", source, numCentroids, B);
        float[] x0 = IOUtils.readFvecsCalcs(new FileInputStream(x0Path), X.length);

        String distToCentroidPath = String.format("%s%s_dist_to_centroid_%d.fvecs", source, dataset, numCentroids);
        float[] distToCentroid = IOUtils.readFvecsCalcs(new FileInputStream(distToCentroidPath), X.length);

        String clusterIdPath = String.format("%s%s_cluster_id_%d.ivecs", source, dataset, numCentroids);
        int[] clusterId = IOUtils.readIvecsCalcs(new FileInputStream(clusterIdPath), X.length);

        String binaryPath = String.format("%sRandNet_C%d_B%d.ivecs", source, numCentroids, B);
        long[][] binary = IOUtils.readBinary(new FileInputStream(binaryPath));

        String indexPath = String.format("%sivfrabitq%d_B%d.index", source, numCentroids, B);
        System.out.println("Loading Complete!");
        IVFRN ivf = new IVFRN(X, centroids, distToCentroid, x0, clusterId, binary);

        System.out.println("Saving");
        ivf.save(indexPath);
    }

    public static void clusterWithIVF(String source, String dataset, int numCentroids) throws IOException {
        Path basePath = Paths.get(source);
        Path fvecPath = Paths.get(basePath.toString(), dataset + ".fvec");

        float[][] X = IOUtils.readFvecs(new FileInputStream(fvecPath.toFile())); // X
        Path centroidsPath = Paths.get(basePath.toString(), dataset + "_centroid_" + numCentroids + ".fvecs");
        Path distToCentroidPath = Paths.get(basePath.toString(), dataset + "_dist_to_centroid_" + numCentroids + ".fvecs");
        Path clusterIdPath = Paths.get(basePath.toString(), dataset + "_cluster_id_" + numCentroids + ".ivecs");

        // cluster data vectors
        System.out.println("cluster data vectors");
        IVF index = new IVF(numCentroids);
        index.train(X);
        Centroid[] centroids = index.getCentroids();
        SearchResult[] results = index.search(X);
        float[] distToCentroids = new float[results.length];
        int[] clusterIds = new int[results.length];
        for(int i = 0; i < results.length; i++) {
            SearchResult result = results[i];
            float distToCentroid = result.distToCentroid();
            clusterIds[i] = result.clusterId();
            distToCentroids[i] = (float) Math.pow(distToCentroid, 0.5);
        }

        IOUtils.toFvecsCalcs(new FileOutputStream(distToCentroidPath.toFile()), distToCentroids);
        IOUtils.toIvecsCalcs(new FileOutputStream(clusterIdPath.toFile()), clusterIds);
        IOUtils.toFvecs(new FileOutputStream(centroidsPath.toFile()), Arrays.stream(centroids).map(Centroid::getVector).toArray(float[][]::new));
    }

    public static void generateSubSpaces(String source, String dataset, int numCentroids) throws IOException {
        // FIXME: FUTURE - clean this usage up
//        String[] datasets = {dataset};
        String path = Paths.get(source).toString();

        // FIXME: FUTURE - support processing multiple datasets
//        for (String dataset : datasets) {
        String dataPath = new File(path, dataset + ".fvec").getAbsolutePath();

        String centroidsPath = new File(path, dataset + "_centroid_" + numCentroids + ".fvecs").getAbsolutePath();
        String clusterIdPath = new File(path, dataset + "_cluster_id_" + numCentroids + ".ivecs").getAbsolutePath();

        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));
        float[][] centroids = IOUtils.readFvecs(new FileInputStream(centroidsPath));
        int[] clusterId = IOUtils.readIvecsCalcs(new FileInputStream(clusterIdPath), X.length);

        int D = X[0].length;
        int B = (D + 63) / 64 * 64;
        int MAX_BD = Math.max(D, B);

        Path projectionPath = Paths.get(new File(path, "P_C" + numCentroids + "_B" + B + ".fvecs").getAbsolutePath());
        Path randomizedCentroidPath = Paths.get(new File(path, "RandCentroid_C" + numCentroids + "_B" + B + ".fvecs").getAbsolutePath());
        Path RNPath = Paths.get(new File(path, "RandNet_C" + numCentroids + "_B" + B + ".ivecs").getAbsolutePath());
        Path x0Path = Paths.get(new File(path, "x0_C" + numCentroids + "_B" + B + ".fvecs").getAbsolutePath());

        float[][] P = getOrthogonalMatrix(MAX_BD);

        // The inverse of an orthogonal matrix equals to its transpose.
        float[][] transposedP = MatrixUtils.transpose(P);

        float[][] XPad = MatrixUtils.padColumns(X, MAX_BD-D);
        float[][] centroidsPad = MatrixUtils.padColumns(centroids, MAX_BD-D);

        float[][] XP = MatrixUtils.dotProduct(XPad, transposedP);
        float[][] CP = MatrixUtils.dotProduct(centroidsPad, transposedP);
        float[][] XP2 = MatrixUtils.subtract(XP, MatrixUtils.indexInto(CP, clusterId));

        boolean[][] binXP = MatrixUtils.greaterThan(XP2, 0);

        // The inner product between the data vector and the quantized data vector
        float[][] XPSubset = MatrixUtils.subset(XP2, B);
        boolean[][] binXPSubset = MatrixUtils.subset(binXP, B);
        float[][] binXPSubsetAsInts = MatrixUtils.asFloats(binXPSubset);
        MatrixUtils.multiplyElementWise(XPSubset, binXPSubsetAsInts);
        MatrixUtils.divide(XPSubset, (float) Math.pow(B, 0.5));
        float[][] x0ri= MatrixUtils.sumRows(XPSubset);
        MatrixUtils.normalize(x0ri, MatrixUtils.normsForRows(XP2), 0.8f);

        long[][] repackedBinXP = MatrixUtils.repackAsUInt64(binXP, B);

        IOUtils.toFvecs(new FileOutputStream(randomizedCentroidPath.toFile()), CP);
        IOUtils.toIvecs(new FileOutputStream(RNPath.toFile()), repackedBinXP);
        IOUtils.toFvecsCalcs(new FileOutputStream(x0Path.toFile()), MatrixUtils.flatten(x0ri));
        IOUtils.toFvecs(new FileOutputStream(projectionPath.toFile()), transposedP);
//        }
    }

    private static float[][] getOrthogonalMatrix(int d) {
        Random random = new Random();
        float[][] G = new float[d][d];
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                G[i][j] = (float) random.nextGaussian();
            }
        }

        QRDecomposition qr = new QRDecomposition(G);
        return qr.getQ();
    }
}

