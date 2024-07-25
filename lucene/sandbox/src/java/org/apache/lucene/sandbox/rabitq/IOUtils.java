package org.apache.lucene.sandbox.rabitq;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

public class IOUtils {

  public static int getTotalFvecs(Path path, int dimensions) throws IOException {
    try (FileInputStream fis = new FileInputStream(path.toFile())) {
      FileChannel fc = fis.getChannel();
      long fsize = fc.size();
      return (int) ((fsize) / (dimensions * 4L + 4L));
    }
  }

  public static float[] fetchFvecsEntry(FileInputStream stream, int dimensions, int vectorIndex)
      throws IOException {
    // FIXME: align along disk boundaries and read in chunks of bytes at a time and then decode then
    // as requested (caching)
    FileChannel fc =
        stream.getChannel(); // FIXME: manage the channel outside of this function for performance??
    fc.position(vectorIndex * (4L + 4L * dimensions));

    ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    fc.read(bb);
    bb.flip();

    float[] data = new float[dimensions];
    bb = ByteBuffer.allocate(dimensions * 4).order(ByteOrder.LITTLE_ENDIAN);
    fc.read(bb);
    bb.flip();
    bb.asFloatBuffer().get(data);

    return data;
  }

  public static float[][] readFvecs(FileInputStream stream) throws IOException {
    FileChannel fc = stream.getChannel();

    ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    fc.read(bb);
    bb.flip();
    int dimensions = bb.getInt();

    long fsize = fc.size();
    int size = (int) ((fsize) / (dimensions * 4 + 4));
    float[][] data = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      if (i != 0) {
        bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        fc.read(bb);
        bb.flip();
        assert dimensions == bb.getInt(); // / dimensions every time
      }

      bb = ByteBuffer.allocate(dimensions * 4).order(ByteOrder.LITTLE_ENDIAN);
      fc.read(bb);
      bb.flip();
      bb.asFloatBuffer().get(data[i]);
    }

    return data;
  }

  public static int[][] readGroundTruth(
      String groundTruthFile, FSDirectory directory, int numQueries) throws IOException {
    if (!Files.exists(directory.getDirectory().resolve(groundTruthFile))) {
      return null;
    }
    int[][] groundTruths = new int[numQueries][];
    // reading the ground truths from the file
    try (IndexInput queryGroundTruthInput =
        directory.openInput(groundTruthFile, IOContext.DEFAULT)) {
      for (int i = 0; i < numQueries; i++) {
        int length = queryGroundTruthInput.readInt();
        groundTruths[i] = new int[length];
        for (int j = 0; j < length; j++) {
          groundTruths[i][j] = queryGroundTruthInput.readInt();
        }
      }
    }
    return groundTruths;
  }

  public static int[][] readIvecs(FileInputStream stream) throws IOException {
    FileChannel fc = stream.getChannel();
    ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    fc.read(bb);
    bb.flip();
    int dimensions = bb.getInt();

    long fsize = fc.size();
    int size = (int) ((fsize) / (dimensions * 4 + 4));
    int[][] data = new int[size][dimensions];
    for (int i = 0; i < size; i++) {
      if (i != 0) {
        bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        fc.read(bb);
        bb.flip();
        assert dimensions == bb.getInt(); // / dimensions every time
      }

      bb = ByteBuffer.allocate(dimensions * 4).order(ByteOrder.LITTLE_ENDIAN);
      fc.read(bb);
      bb.flip();
      bb.asIntBuffer().get(data[i]);
    }

    return data;
  }

  public static void toFvecs(FileOutputStream stream, float[][] data) throws IOException {
    FileChannel fc = stream.getChannel();
    int dimensions = data[0].length;
    ByteBuffer bb;

    for (int i = 0; i < data.length; i++) {
      bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      bb.putInt(data[0].length);
      bb.flip();
      fc.write(bb);

      bb = ByteBuffer.allocate(dimensions * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (float d : data[i]) {
        bb.putFloat(d);
      }
      bb.flip();
      fc.write(bb);
    }
  }

  public static void toIvecs(FileOutputStream stream, int[][] data) throws IOException {
    FileChannel fc = stream.getChannel();
    int dimensions = data[0].length;
    ByteBuffer bb;

    for (int i = 0; i < data.length; i++) {
      bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      bb.putInt(data[0].length);
      bb.flip();
      fc.write(bb);

      bb = ByteBuffer.allocate(dimensions * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (int d : data[i]) {
        bb.putInt(d);
      }
      bb.flip();
      fc.write(bb);
    }
  }
}
