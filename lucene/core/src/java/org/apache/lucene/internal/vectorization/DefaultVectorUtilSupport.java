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

package org.apache.lucene.internal.vectorization;

import java.util.List;
import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.SuppressForbidden;

final class DefaultVectorUtilSupport implements VectorUtilSupport {

  DefaultVectorUtilSupport() {}

  // the way FMA should work! if available use it, otherwise fall back to mul/add
  @SuppressForbidden(reason = "Uses FMA only where fast and carefully contained")
  private static float fma(float a, float b, float c) {
    if (Constants.HAS_FAST_SCALAR_FMA) {
      return Math.fma(a, b, c);
    } else {
      return a * b + c;
    }
  }

  @Override
  public float dotProduct(float[] a, float[] b) {
    float res = 0f;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;
      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        acc1 = fma(a[i], b[i], acc1);
        acc2 = fma(a[i + 1], b[i + 1], acc2);
        acc3 = fma(a[i + 2], b[i + 2], acc3);
        acc4 = fma(a[i + 3], b[i + 3], acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    for (; i < a.length; i++) {
      res = fma(a[i], b[i], res);
    }
    return res;
  }

  @Override
  public float cosine(float[] a, float[] b) {
    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float sum1 = 0;
      float sum2 = 0;
      float norm1_1 = 0;
      float norm1_2 = 0;
      float norm2_1 = 0;
      float norm2_2 = 0;

      int upperBound = a.length & ~(2 - 1);
      for (; i < upperBound; i += 2) {
        // one
        sum1 = fma(a[i], b[i], sum1);
        norm1_1 = fma(a[i], a[i], norm1_1);
        norm2_1 = fma(b[i], b[i], norm2_1);

        // two
        sum2 = fma(a[i + 1], b[i + 1], sum2);
        norm1_2 = fma(a[i + 1], a[i + 1], norm1_2);
        norm2_2 = fma(b[i + 1], b[i + 1], norm2_2);
      }
      sum += sum1 + sum2;
      norm1 += norm1_1 + norm1_2;
      norm2 += norm2_1 + norm2_2;
    }

    for (; i < a.length; i++) {
      sum = fma(a[i], b[i], sum);
      norm1 = fma(a[i], a[i], norm1);
      norm2 = fma(b[i], b[i], norm2);
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public float squareDistance(float[] a, float[] b) {
    float res = 0;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;

      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        // one
        float diff1 = a[i] - b[i];
        acc1 = fma(diff1, diff1, acc1);

        // two
        float diff2 = a[i + 1] - b[i + 1];
        acc2 = fma(diff2, diff2, acc2);

        // three
        float diff3 = a[i + 2] - b[i + 2];
        acc3 = fma(diff3, diff3, acc3);

        // four
        float diff4 = a[i + 3] - b[i + 3];
        acc4 = fma(diff4, diff4, acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    for (; i < a.length; i++) {
      float diff = a[i] - b[i];
      res = fma(diff, diff, res);
    }
    return res;
  }

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += a[i] * b[i];
    }
    return total;
  }

  @Override
  public int int4DotProduct(byte[] a, boolean apacked, byte[] b, boolean bpacked) {
    assert (apacked && bpacked) == false;
    if (apacked || bpacked) {
      byte[] packed = apacked ? a : b;
      byte[] unpacked = apacked ? b : a;
      int total = 0;
      for (int i = 0; i < packed.length; i++) {
        byte packedByte = packed[i];
        byte unpacked1 = unpacked[i];
        byte unpacked2 = unpacked[i + packed.length];
        total += (packedByte & 0x0F) * unpacked2;
        total += ((packedByte & 0xFF) >> 4) * unpacked1;
      }
      return total;
    }
    return dotProduct(a, b);
  }

  @Override
  public float cosine(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;

    for (int i = 0; i < a.length; i++) {
      byte elem1 = a[i];
      byte elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public int squareDistance(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int squareSum = 0;
    for (int i = 0; i < a.length; i++) {
      int diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  @Override
  public int findNextGEQ(int[] buffer, int target, int from, int to) {
    for (int i = from; i < to; ++i) {
      if (buffer[i] >= target) {
        return i;
      }
    }
    return to;
  }

  @Override
  public long int4BitDotProduct(byte[] int4Quantized, byte[] binaryQuantized) {
    return int4BitDotProductImpl(int4Quantized, binaryQuantized);
  }

  @Override
  public float calculateOSQLoss(
      float[] target, float[] interval, float step, float invStep, float norm2, float lambda) {
    float a = interval[0];
    float b = interval[1];
    float xe = 0f;
    float e = 0f;
    for (float xi : target) {
      // this is quantizing and then dequantizing the vector
      float xiq = fma(step, Math.round((Math.min(Math.max(xi, a), b) - a) * invStep), a);
      // how much does the de-quantized value differ from the original value
      float xiiq = xi - xiq;
      e = fma(xiiq, xiiq, e);
      xe = fma(xi, xiiq, xe);
    }
    return (1f - lambda) * xe * xe / norm2 + lambda * e;
  }

  @Override
  public void calculateOSQGridPoints(
      float[] target, float[] interval, int points, float invStep, float[] pts) {
    float a = interval[0];
    float b = interval[1];
    float daa = 0;
    float dab = 0;
    float dbb = 0;
    float dax = 0;
    float dbx = 0;
    for (float v : target) {
      float k = Math.round((Math.min(Math.max(v, a), b) - a) * invStep);
      float s = k / (points - 1);
      float ms = 1f - s;
      daa = fma(ms, ms, daa);
      dab = fma(ms, s, dab);
      dbb = fma(s, s, dbb);
      dax = fma(ms, v, dax);
      dbx = fma(s, v, dbx);
    }
    pts[0] = daa;
    pts[1] = dab;
    pts[2] = dbb;
    pts[3] = dax;
    pts[4] = dbx;
  }

  @Override
  public void centerAndCalculateOSQStatsEuclidean(
      float[] vector, float[] centroid, float[] centered, float[] stats) {
    float vecMean = 0;
    float vecVar = 0;
    float norm2 = 0;
    float min = Float.MAX_VALUE;
    float max = -Float.MAX_VALUE;
    for (int i = 0; i < vector.length; i++) {
      centered[i] = vector[i] - centroid[i];
      min = Math.min(min, centered[i]);
      max = Math.max(max, centered[i]);
      norm2 = fma(centered[i], centered[i], norm2);
      float delta = centered[i] - vecMean;
      vecMean += delta / (i + 1);
      vecVar = fma(delta, (centered[i] - vecMean), vecVar);
    }
    stats[0] = vecMean;
    stats[1] = vecVar / vector.length;
    stats[2] = norm2;
    stats[3] = min;
    stats[4] = max;
  }

  @Override
  public void centerAndCalculateOSQStatsDp(
      float[] vector, float[] centroid, float[] centered, float[] stats) {
    float vecMean = 0;
    float vecVar = 0;
    float norm2 = 0;
    float centroidDot = 0;
    float min = Float.MAX_VALUE;
    float max = -Float.MAX_VALUE;
    for (int i = 0; i < vector.length; i++) {
      centroidDot = fma(vector[i], centroid[i], centroidDot);
      centered[i] = vector[i] - centroid[i];
      min = Math.min(min, centered[i]);
      max = Math.max(max, centered[i]);
      norm2 = fma(centered[i], centered[i], norm2);
      float delta = centered[i] - vecMean;
      vecMean += delta / (i + 1);
      float delta2 = centered[i] - vecMean;
      vecVar = fma(delta, delta2, vecVar);
    }
    stats[0] = vecMean;
    stats[1] = vecVar / vector.length;
    stats[2] = norm2;
    stats[3] = min;
    stats[4] = max;
    stats[5] = centroidDot;
  }

  @Override
  public void calculateCentroid(List<float[]> vectors, float[] centroid) {
    calculateCentroidImpl(vectors, centroid);
  }

  public static void calculateCentroidImpl(List<float[]> vectors, float[] centroid) {
    for (var vector : vectors) {
      assert vector.length == centroid.length;
      for (int i = 0; i < centroid.length; i++) {
        centroid[i] += vector[i];
      }
    }
    for (int i = 0; i < centroid.length; i++) {
      centroid[i] /= vectors.size();
    }
  }

  public static long int4BitDotProductImpl(byte[] q, byte[] d) {
    assert q.length == d.length * 4;
    long ret = 0;
    int size = d.length;
    for (int i = 0; i < 4; i++) {
      int r = 0;
      long subRet = 0;
      for (final int upperBound = d.length & -Integer.BYTES; r < upperBound; r += Integer.BYTES) {
        subRet +=
            Integer.bitCount(
                (int) BitUtil.VH_NATIVE_INT.get(q, i * size + r)
                    & (int) BitUtil.VH_NATIVE_INT.get(d, r));
      }
      for (; r < d.length; r++) {
        subRet += Integer.bitCount((q[i * size + r] & d[r]) & 0xFF);
      }
      ret += subRet << i;
    }
    return ret;
  }
}
