package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.vectorization.OSQVectorsScorer;
import org.apache.lucene.internal.vectorization.VectorizationProvider;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
// first iteration is complete garbage, so make sure we really warmup
@Warmup(iterations = 4, time = 1)
// real iterations. not useful to spend tons of time here, better to fork more
@Measurement(iterations = 5, time = 1)
// engage some noise reduction
@Fork(value = 1)
public class OSQScorerBenchmark {

  @Param({"1024"})
  int dims;

  int length;

  int numVectors = OSQVectorsScorer.BULK_SIZE * 10;
  int numQueries = 10;

  byte[][] binaryVectors;
  byte[][] binaryQueries;
  OptimizedScalarQuantizer.QuantizationResult result;
  float centroidDp;

  byte[] scratch;
  OSQVectorsScorer scorer;

  IndexInput in;

  float[] scratchScores;
  float[] corrections;

  @Setup
  public void setup() throws IOException {
    Random random = new Random(123);

    this.length = OptimizedScalarQuantizer.discretize(dims, 64) / 8;

    binaryVectors = new byte[numVectors][length];
    for (byte[] binaryVector : binaryVectors) {
      random.nextBytes(binaryVector);
    }

    Directory dir = new MMapDirectory(Files.createTempDirectory("vectorData"));
    IndexOutput out = dir.createOutput("vectors", IOContext.DEFAULT);
    byte[] correctionBytes = new byte[14 * OSQVectorsScorer.BULK_SIZE];
    for (int i = 0; i < numVectors; i += OSQVectorsScorer.BULK_SIZE) {
      for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
        out.writeBytes(binaryVectors[i + j], 0, binaryVectors[i + j].length);
      }
      random.nextBytes(correctionBytes);
      out.writeBytes(correctionBytes, 0, correctionBytes.length);
    }
    out.close();
    in = dir.openInput("vectors", IOContext.DEFAULT);

    binaryQueries = new byte[numVectors][4 * length];
    for (byte[] binaryVector : binaryVectors) {
      random.nextBytes(binaryVector);
    }
    result =
        new OptimizedScalarQuantizer.QuantizationResult(
            random.nextFloat(),
            random.nextFloat(),
            random.nextFloat(),
            Short.toUnsignedInt((short) random.nextInt()));
    centroidDp = random.nextFloat();

    scratch = new byte[length];
    scorer = VectorizationProvider.getInstance().newOSQVectorsScorer(in, dims);
    scratchScores = new float[16];
    corrections = new float[3];
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void scoreFromArray(Blackhole bh) throws IOException {
    for (int j = 0; j < numQueries; j++) {
      in.seek(0);
      for (int i = 0; i < numVectors; i++) {
        in.readBytes(scratch, 0, length);
        float qDist = VectorUtil.int4BitDotProduct(binaryQueries[j], scratch);
        in.readFloats(corrections, 0, corrections.length);
        int addition = Short.toUnsignedInt(in.readShort());
        float score =
            scorer.score(
                result,
                VectorSimilarityFunction.EUCLIDEAN,
                centroidDp,
                corrections[0],
                corrections[1],
                addition,
                corrections[2],
                qDist);
        bh.consume(score);
      }
    }
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void scoreFromMemorySegmentOnlyVector(Blackhole bh) throws IOException {
    for (int j = 0; j < numQueries; j++) {
      in.seek(0);
      for (int i = 0; i < numVectors; i++) {
        float qDist = scorer.quantizeScore(binaryQueries[j]);
        in.readFloats(corrections, 0, corrections.length);
        int addition = Short.toUnsignedInt(in.readShort());
        float score =
            scorer.score(
                result,
                VectorSimilarityFunction.EUCLIDEAN,
                centroidDp,
                corrections[0],
                corrections[1],
                addition,
                corrections[2],
                qDist);
        bh.consume(score);
      }
    }
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void scoreFromMemorySegmentOnlyVectorBulk(Blackhole bh) throws IOException {
    for (int j = 0; j < numQueries; j++) {
      in.seek(0);
      for (int i = 0; i < numVectors; i += 16) {
        scorer.quantizeScoreBulk(binaryQueries[j], OSQVectorsScorer.BULK_SIZE, scratchScores);
        for (int k = 0; k < OSQVectorsScorer.BULK_SIZE; k++) {
          in.readFloats(corrections, 0, corrections.length);
          int addition = Short.toUnsignedInt(in.readShort());
          float score =
              scorer.score(
                  result,
                  VectorSimilarityFunction.EUCLIDEAN,
                  centroidDp,
                  corrections[0],
                  corrections[1],
                  addition,
                  corrections[2],
                  scratchScores[k]);
          bh.consume(score);
        }
      }
    }
  }

  @Benchmark
  @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
  public void scoreFromMemorySegmentAllBulk(Blackhole bh) throws IOException {
    for (int j = 0; j < numQueries; j++) {
      in.seek(0);
      for (int i = 0; i < numVectors; i += 16) {
        scorer.scoreBulk(
            binaryQueries[j],
            result,
            VectorSimilarityFunction.EUCLIDEAN,
            centroidDp,
            scratchScores);
        bh.consume(scratchScores);
      }
    }
  }
}
