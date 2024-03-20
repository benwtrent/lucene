package org.apache.lucene.util.quantization;

public class BinaryQuantizer {
  private final float quantizationMedian;

  public BinaryQuantizer() {
    this.quantizationMedian = 0;
  }

  public BinaryQuantizer(float quantizationMedian) {
    this.quantizationMedian = quantizationMedian;
  }

  public void quantize(float[] vector, byte[] destination) {
    assert vector.length % 8 == 0;
    assert destination.length == vector.length >> 3;
    for (int i = 0; i < vector.length; i++) {
      int byteIndex = i >> 3;
      int bitIndex = i & 7;
      float value = vector[i];
      if (value > quantizationMedian) {
        destination[byteIndex] |= (byte) (1 << bitIndex);
      } else {
        destination[byteIndex] &= (byte) ~(1 << bitIndex);
      }
    }
  }

  public void quantizeNoCompression(float[] vector, byte[] destination) {
    assert vector.length % 8 == 0;
    assert destination.length == vector.length;
    for (int i = 0; i < vector.length; i++) {
      float value = vector[i];
      destination[i] = (byte) (value > quantizationMedian ? 1 : 0);
    }
  }
}
