package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;

public class Search {

    public static void main(String[] args) throws Exception {
        // FIXME: FUTURE - use fastscan to search over the ivfrn?
        // FIXME: FUTURE - get metrics setup appropriately as needed
        // FIXME: FUTURE - better arg parsing

        // search DIRECTORY_TO_DATASET DATASET_NAME NUM_CENTROIDS DIMENSIONS B_QUERY OUTPUT_PATH
        String source = "/Users/benjamintrent/rabit_data/"; ;//args[0];  // eg "/Users/jwagster/Desktop/gist1m/gist/"
        String dataset = "quora-522k-e5small_corpus-quora-E5-small"; //args[1]; // eg "gist"
        int numCentroids = 1;//Integer.parseInt(args[2]);
        int dimensions = 384;// Integer.parseInt(args[3]);
        int B_QUERY = 4;//Integer.parseInt(args[4]);
        int k = 10;//Integer.parseInt(args[5]);

        // FIXME: FUTURE - clean up these constants
        int B = (dimensions + 63) / 64 * 64;

        // FIXME: FUTURE - clean up gross path mgmt
        String queryPath = String.format("%s%s", source, "quora-522k-e5small_queries-quora-E5-small.fvec");
        float[][] Q = IOUtils.readFvecs(new FileInputStream(queryPath));

        String dataPath = String.format("%s%s.fvec", source, dataset);
        float[][] X = IOUtils.readFvecs(new FileInputStream(dataPath));

        String groundTruthPath = String.format("%s%s", source, "quora-522k-e5small_groundtruth-quora-E5-small.ivec");
        int[][] G = IOUtils.readIvecs(new FileInputStream(groundTruthPath));

        String transformationPath = String.format("%sP_C%d_B%d.fvecs", source, numCentroids, B);
        float[][] P = IOUtils.readFvecs(new FileInputStream(transformationPath));

        String indexPath = String.format("%sivfrabitq%d_B%d.index", source, numCentroids, B);

        IVFRN ivfrn = IVFRN.load(indexPath);
        float[][] RandQ = MatrixUtils.multiply(Q, P);
        // warmup
        System.out.println("Warmup");
        test(Q, 100, RandQ, X, G, ivfrn, k, B_QUERY);

        System.out.println("Actually benchmarking");
        test(Q, 100, RandQ, X, G, ivfrn, k, B_QUERY);
    }

    public static void test(float[][] Q, float[][] RandQ, float[][] X, int[][] G, IVFRN ivf, int k, int B_QUERY) {
        test(Q, Q.length, RandQ, X, G, ivf, k, B_QUERY);
    }

    public static void test(float[][] Q, int numQ, float[][] RandQ, float[][] X, int[][] G, IVFRN ivf, int k, int B_QUERY) {

        int nprobes = 300;
        nprobes = Math.min(nprobes, ivf.C); // FIXME: FUTURE - hardcoded
        assert nprobes <= k;

        long totalUsertime = 0;
        int correctCount = 0;

        System.out.println("Starting search");
        for (int i = 0; i < numQ; i++) {
            long startTime = System.nanoTime();
            IVFRNResult result = ivf.search(Q[i], RandQ[i], k, nprobes, B_QUERY);
            PriorityQueue<Result> KNNs = result.results();
            totalUsertime +=  System.nanoTime() - startTime;

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
            // FIXME: FUTURE - use logging instead
//            System.out.println("recall = " + correct + " / " + k + " " + (i + 1) + " / " + Q.length + " " + usertime + " us" + " err bound avg = " + stats.errorBoundAvg() + " nn explored = " + stats.totalExploredNNs());

/*            if (i % 1500 == 0) {
                System.out.print(".");
            }*/
        }
        System.out.println();
        float totalUsTime = TimeUnit.NANOSECONDS.toMicros(totalUsertime);

        // FIXME: FUTURE - missing rotation time?
        float timeUsPerQuery = totalUsTime / numQ;
        float recall = (float) correctCount / (numQ * k);

        // FIXME: FUTURE - logs instead of println
        System.out.println("------------------------------------------------");
        System.out.println("nprobe = " + nprobes + "\tk = " + k + "\tCoarse Clusters = " + ivf.centroids.length);
        System.out.println("Recall = " + recall * 100f);
        System.out.println("Avg Time Per Search = " + timeUsPerQuery + " us\t QPS = " + (1e6 / timeUsPerQuery) + " query/s");
        System.out.println("Total Search Time = " + TimeUnit.NANOSECONDS.toSeconds(totalUsertime) + " sec");
    }

}
