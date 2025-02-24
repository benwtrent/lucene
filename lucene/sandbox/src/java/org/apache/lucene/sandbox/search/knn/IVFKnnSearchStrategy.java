/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.apache.lucene.sandbox.search.knn;

import java.util.List;
import java.util.Objects;
import org.apache.lucene.search.knn.KnnSearchStrategy;

public class IVFKnnSearchStrategy extends KnnSearchStrategy {
  private final int nProbe;
  private final List<Integer> centroids;

  public IVFKnnSearchStrategy(int nProbe) {
    this.nProbe = nProbe;
    this.centroids = null;
  }

  public IVFKnnSearchStrategy(int nProbe, List<Integer> centroids) {
    this.nProbe = nProbe;
    this.centroids = centroids;
  }

  public List<Integer> getCentroids() {
    return centroids;
  }

  public int getNProbe() {
    return nProbe;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    IVFKnnSearchStrategy that = (IVFKnnSearchStrategy) o;
    return nProbe == that.nProbe;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(nProbe);
  }
}
