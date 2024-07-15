package org.apache.lucene.sandbox.rabitq;

import org.apache.lucene.util.ArrayUtil;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.PriorityQueue;


public class IVFRN {
    public Factor[] fac;

    public int N;                        // the number of data vectors
    public int C;                        // the number of clusters

    public int[] start;                  // the start point of a cluster
    public int[] len;                    // the length of a cluster
    public int[] id;                     // N of size_t the ids of the objects in a cluster
    public float[] distToC;              // N of floats distance to the centroids (not the squared distance)
    public float[] u;                    // B of floats random numbers sampled from the uniform distribution [0,1]

    // FIXME: FUTURE - make this a byte[] instead??
    public long[][] binaryCode;          // (B / 64) * N of 64-bit uint64_t
    public byte[][] binaryCodeByte;          // (B / 64) * N of 64-bit uint64_t

    public float[] x0;                   // N of floats in the Random Net algorithm
    public float[][] centroids;          // N * B floats (not N * D), note that the centroids should be randomized
    public float[][] data;               // N * D floats, note that the datas are not randomized

    private final int B;
    private final int D;

    public IVFRN(float[][] X, float[][] centroids, float[] distToCentroid, float[] _x0, int[] clusterId, long[][] binary) {
        D = X[0].length;
        B = (D + 63) / 64 * 64;

        N = X.length;
        C = centroids.length;

        // Check if B is greater than or equal to D
        assert (B >= D);

        start = new int[C];
        len = new int[C];
        id = new int[N];
        distToC = new float[N];
        x0 = new float[N];

        for (int i = 0; i < N; i++) {
            len[clusterId[i]]++;
        }
        int sum = 0;
        for (int i = 0; i < C; i++) {
            start[i] = sum;
            sum += len[i];
        }

        for (int i = 0; i < N; i++) {
            id[start[clusterId[i]]] = i;
            distToC[start[clusterId[i]]] = distToCentroid[i];
            x0[start[clusterId[i]]] = _x0[i];
            start[clusterId[i]]++;
        }

        for (int i = 0; i < C; i++) {
            start[i] -= len[i];
        }

        this.centroids = centroids;

        this.data = new float[N][X[0].length];
        this.binaryCode = new long[N][B/64];
        for(int i = 0; i < N; i++) {
            int x = id[i];
            data[i] = X[x];
            binaryCode[i] = binary[x];
        }
    }

    private IVFRN(int n, int d, int c, int b, float[][] centroids, float[][] data, long[][] binaryCode, int[] start,
                  int[] len, int[] id, float[] distToC, float[] x0, float[] u, Factor[] fac) {
        this.N = n;
        this.D = d;
        this.C = c;
        this.B = b;

        this.centroids = centroids;
        this.data = data;
        this.binaryCodeByte = new byte[N][];
        this.binaryCode = binaryCode;
        for(int i = 0; i < N; i++) {
            ByteBuffer buf = ByteBuffer.allocate(B/8);
            buf.order(ByteOrder.LITTLE_ENDIAN);
            for(long val : binaryCode[i]) {
                buf.putLong(val);
            }
            this.binaryCodeByte[i] = buf.array();
        }
        this.start = start;
        this.len = len;
        this.id = id;
        this.distToC = distToC;
        this.x0 = x0;
        this.u = u;
        this.fac = fac;
    }

    public void save(String filename) throws IOException {
        try(FileOutputStream fos = new FileOutputStream(filename); FileChannel fc = fos.getChannel()) {
            ByteBuffer bb = ByteBuffer.allocate(4*4).order(ByteOrder.LITTLE_ENDIAN);
            bb.putInt(N);
            bb.putInt(D);
            bb.putInt(C);
            bb.putInt(B);
            bb.flip();
            fc.write(bb);

            //start, len, id, distToC, x0, centroids, data, binaryCode
            bb = ByteBuffer.allocate(4*C + 4*C + 4*N + 4*N + 4*N + 4*C*B + 4*N*D + 8*N*B/64).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < C; i++) {
                bb.putInt(start[i]);
            }

            for (int i = 0; i < C; i++) {
                bb.putInt(len[i]);
            }

            for (int i = 0; i < N; i++) {
                bb.putInt(id[i]);
            }

            for (int i = 0; i < N; i++) {
                bb.putFloat(distToC[i]);
            }

            for (int i = 0; i < N; i++) {
                bb.putFloat(x0[i]);
            }

            for (int i = 0; i < C; i++) {
                for (int j = 0; j < B; j++) {
                    bb.putFloat(centroids[i][j]);
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    bb.putFloat(data[i][j]);
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < B / 64; j++) {
                    bb.putLong(binaryCode[i][j]);
                }
            }

            bb.flip();
            fc.write(bb);
        }
    }

    public static IVFRN load(String filename) throws IOException {
        System.out.println("Loading " + filename);
        try(FileInputStream fis = new FileInputStream(filename); FileChannel fc = fis.getChannel()) {
            ByteBuffer bb = ByteBuffer.allocate(4*4).order(ByteOrder.LITTLE_ENDIAN);
            fc.read(bb);
            bb.flip();
            int N = bb.getInt();
            int D = bb.getInt();
            int C = bb.getInt();
            int B = bb.getInt();

            float fac_norm = (float) Utils.constSqrt(1.0 * B);
            float max_x1 = (float) (1.9 / Utils.constSqrt(1.0 * B-1.0));

            float[][] centroids = new float[C][B];
            float[][] data = new float[N][D];

            long[][] binaryCode = new long[N][B / 64];
            int[] start = new int[C];
            int[] len = new int[C];
            int[] id = new int[N];
            float[] distToC = new float[N];
            float[] x0 = new float[N];

            //start, len, id, distToC, x0, centroids, data, binaryCode
            bb = ByteBuffer.allocate(4*C + 4*C + 4*N + 4*N + 4*N + 4*C*B + 4*N*D + 8*N*B/64).order(ByteOrder.LITTLE_ENDIAN);
            fc.read(bb);
            bb.flip();

            for (int i = 0; i < C; i++) {
                start[i] = bb.getInt();
            }

            for (int i = 0; i < C; i++) {
                len[i] = bb.getInt();
            }

            for (int i = 0; i < N; i++) {
                id[i] = bb.getInt();
            }

            for (int i = 0; i < N; i++) {
                distToC[i] = bb.getFloat();
            }

            for (int i = 0; i < N; i++) {
                x0[i] = bb.getFloat();
            }

            for (int i = 0; i < C; i++) {
                for (int j = 0; j < B; j++) {
                    centroids[i][j] = bb.getFloat();
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    data[i][j] = bb.getFloat();
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < B / 64; j++) {
                    binaryCode[i][j] = bb.getLong();
                }
            }

            float[] u = new float[B];
            for (int i = 0; i < B; i++) {
                u[i] = (float) Math.random();
            }

            Factor[] fac = new Factor[N];
            for (int i = 0; i < N; i++) {
                double x_x0 = distToC[i] / x0[i];
                float sqrX = distToC[i] * distToC[i];
                float error = (float) (2.0 * max_x1 * Math.sqrt(x_x0 * x_x0 - distToC[i] * distToC[i]));
                float factorPPC = (float) (-2.0 / fac_norm * x_x0 * ((float) SpaceUtils.popcount(binaryCode[i], B) * 2.0 - B));
                float factorIP = (float) (-2.0 / fac_norm * x_x0);
                fac[i] = new Factor(sqrX, error, factorPPC, factorIP);
            }

            return new IVFRN(N, D, C, B, centroids, data, binaryCode, start, len, id, distToC, x0, u, fac);
        }
    }

    @SuppressWarnings("unchecked")
    public IVFRNResult search(float[] query, float[] rdQuery, int k, int nProbe, int B_QUERY) {
        //FIXME: FUTURE - implement fast scan and do a comparison

        assert nProbe < C;

        PriorityQueue<Result> knns = new PriorityQueue<>(k, Comparator.reverseOrder());

        // Find out the nearest N_{probe} centroids to the query vector.
        PriorityQueue<Result> topNProbeCentroids = new PriorityQueue<>(nProbe);
        for (int i = 0; i < C; i++) {
            topNProbeCentroids.add(new Result(VectorUtils.squareDistance(rdQuery, centroids[i]), i));
        }

        Result[] centroidDist = new Result[C];
        for(int i = 0; i < nProbe; i++) {
            centroidDist[i] = topNProbeCentroids.remove();
        }

        // FIXME: FUTURE - don't use the Result class for this; it's confusing
        int maxEstimatorSize = k;
        PriorityQueue<Result> estimatorDistances = new PriorityQueue<>(maxEstimatorSize, Comparator.reverseOrder());

        for (int pb = 0; pb < nProbe; pb++) {
            int c = centroidDist[pb].c();
            float sqrY = centroidDist[pb].sqrY();

            if(!Float.isFinite(sqrY)) {
                continue;
            }

            // Preprocess the residual query and the quantized query
            float[] v = SpaceUtils.range(rdQuery, centroids[c]);
            float vl = v[0], vr = v[1];
            float width = (vr - vl) / ((1 << B_QUERY) - 1);

            QuantResult quantResult = SpaceUtils.quantize(rdQuery, centroids[c], u, vl, width);
            byte[] byteQuery = quantResult.result();
            int sumQ = quantResult.sumQ();

            // Binary String Representation
            byte[] quantQuery = SpaceUtils.transposeBin(byteQuery, D, B_QUERY);

            int startC = start[c];

            int len_sub = len[c];

          float y = (float) Math.sqrt(sqrY);

            int facCounter = startC;
            int bCounter = startC;

            for (int i = 0; i < len_sub; i++) {
                float tmpDist = fac[facCounter].sqrX() + sqrY + fac[facCounter].factorPPC() * vl +
                        (SpaceUtils.ipByteBin(quantQuery, binaryCodeByte[bCounter], B_QUERY, B) * 2 - (float) sumQ) *
                                fac[facCounter].factorIP() * width;
                float errorBound = y * (fac[facCounter].error());
                float estimator = tmpDist - errorBound;

                if (estimatorDistances.size() < maxEstimatorSize) {
                    estimatorDistances.add(new Result(estimator, startC + i));
                } else if (estimator < estimatorDistances.peek().sqrY()) {
                    estimatorDistances.poll();
                    estimatorDistances.add(new Result(estimator, startC + i));
                }

                bCounter++;
                facCounter++;
            }
        }

        int size = estimatorDistances.size();
        if (size == k) {
            return new IVFRNResult(estimatorDistances, null);
        }
        for(int i = 0; i < size; i++) {
            Result res = estimatorDistances.poll();
            float actualDist = VectorUtils.squareDistance(query, data[res.c()]);
            if (knns.size() < k) {
                knns.add(new Result(actualDist, id[res.c()]));
            } else if (actualDist < knns.peek().sqrY()) {
                knns.poll();
                knns.add(new Result(actualDist, id[res.c()]));
            }
        }

        return new IVFRNResult(knns, null);
    }
}

