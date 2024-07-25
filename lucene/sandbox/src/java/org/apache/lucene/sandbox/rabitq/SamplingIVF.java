package org.apache.lucene.sandbox.rabitq;

import java.io.IOException;
import java.util.Arrays;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

public class SamplingIVF {
  private static final int seed = 1;

  private final int totalClusters;
  private Centroid[] centroids;

  public SamplingIVF(int clusters) {
    this.totalClusters = clusters;
  }

  public void train(RandomAccessVectorValues.Floats vectors, int dimensions) throws IOException {
    // FIXME: FUTURE do error checking ... has to be at least as many vectors as centroids for now
    SampleReader sampleReader = SampleReader.createSampleReader(vectors, 131072, seed);
    KMeans kMeans = new KMeans(sampleReader, totalClusters, seed);
    float[][] centroids = kMeans.computeCentroids(5, 10, f -> {});

    // FIXME: build this random selection of of the first N vectors where N is a reasonable set to
    // consider
    // randomly create totalClusters centroids from existing vectors
    this.centroids = new Centroid[totalClusters];
    for (int i = 0; i < totalClusters; i++) {
      this.centroids[i] = new Centroid(i, Arrays.copyOf(centroids[i], dimensions));
    }
  }

  public Centroid[] getCentroids() {
    return centroids;
  }

  public SearchResult[] search(RandomAccessVectorValues.Floats vectorValues) throws IOException {

    SearchResult[] searchResults = new SearchResult[vectorValues.size()];

    for (int i = 0; i < vectorValues.size(); i++) {
      float[] vector = vectorValues.vectorValue(i);

      float smallestDToCentroid = Float.MAX_VALUE;
      int centroid = -1;
      for (int j = 0; j < centroids.length; j++) {
        float d = VectorUtils.squareDistance(centroids[j].getVector(), vector);
        if (d < smallestDToCentroid) {
          smallestDToCentroid = d;
          centroid = j;
        }
      }
      searchResults[i] = new SearchResult(smallestDToCentroid, centroids[centroid].getId());
    }
    return searchResults;
  }
}
