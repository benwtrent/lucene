package org.apache.lucene.sandbox.rabitq;

import java.util.BitSet;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.util.BitUtil;

public class SpaceUtils {

  private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Byte> BYTE_128_SPECIES = ByteVector.SPECIES_128;
  private static final VectorSpecies<Integer> INT_SPECIES;


  static final int VECTOR_BITSIZE;
  static {
    // default to platform supported bitsize
    int vectorBitSize = VectorShape.preferredShape().vectorBitSize();
    // but allow easy overriding for testing
    INT_SPECIES = VectorSpecies.of(int.class, VectorShape.forBitSize(vectorBitSize));
    VECTOR_BITSIZE = INT_SPECIES.vectorBitSize();
  }

  public static final int B_QUERY = 4;

  public static int popcount(byte[] d, int B) {
    return BitSet.valueOf(d).cardinality();
  }

  public static long ipByteBinByte(byte[] q, byte[] d) {
    long ret = 0;
    int size = d.length;
    for (int i = 0; i < B_QUERY; i++) {
      int r = 0;
      long subRet = 0;
      for (final int upperBound = d.length & -Integer.BYTES; r < upperBound; r += Integer.BYTES) {
        subRet +=
            Integer.bitCount(
                (int) BitUtil.VH_NATIVE_INT.get(q, i * size + r)
                    & (int) BitUtil.VH_NATIVE_INT.get(d, r));
      }
      for (; r < d.length; r++) {
        subRet += Integer.bitCount((q[i * size + r] & d[i]) & 0xFF);
      }
      ret += subRet << i;
    }
    return ret;
  }

  public static long bitAnd(byte[] q, byte[] d) {
    long ret = 0;
    final int limit = BYTE_128_SPECIES.loopBound(d.length);
    for (int j = 0; j < limit; j += 256) {
      ByteVector acc = ByteVector.zero(BYTE_128_SPECIES);
      int innerLimit = Math.min(limit - j, 256);
      for (int k = 0; k < innerLimit; k += BYTE_128_SPECIES.length()) {
        ByteVector vd = ByteVector.fromArray(BYTE_128_SPECIES, d, j + k);
        ByteVector vq = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k);
        ByteVector vres = vq.and(vd);
        acc = acc.add(vres);
      }
      ShortVector sumShort1 = acc.reinterpretAsShorts().and((short) 0xFF);
      ShortVector sumShort2 = acc.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      ret += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);
    }
    // tail as bytes
    for (int r = limit; r < d.length; r++) {
      ret += Integer.bitCount((q[r] & d[r]) & 0xFF);
    }
    return ret;
  }

  public static long ipByteBinBytePanbq2(byte[] q, byte[] d) {
    long ret = 0;
    long subRet0 = 0;
    long subRet1 = 0;

    // TODO This is best for ARM, need to test on 256 & 512
    final int limit = BYTE_128_SPECIES.loopBound(d.length);
    // iterate in chunks of 256 bytes to ensure we don't overflow the accumulator
    // (256bytes/16lanes=16itrs)
    for (int j = 0; j < limit; j += 256) {
      ByteVector acc0 = ByteVector.zero(BYTE_128_SPECIES);
      ByteVector acc1 = ByteVector.zero(BYTE_128_SPECIES);
      int innerLimit = Math.min(limit - j, 256);
      for (int k = 0; k < innerLimit; k += BYTE_128_SPECIES.length()) {
        ByteVector vd = ByteVector.fromArray(BYTE_128_SPECIES, d, j + k);
        ByteVector vq0 = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k);
        ByteVector vq1 = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k + d.length);
        ByteVector vres0 = vq0.and(vd);
        ByteVector vres1 = vq1.and(vd);
        vres0 = vres0.lanewise(VectorOperators.BIT_COUNT);
        vres1 = vres1.lanewise(VectorOperators.BIT_COUNT);
        acc0 = acc0.add(vres0);
        acc1 = acc1.add(vres1);
      }
      ShortVector sumShort1 = acc0.reinterpretAsShorts().and((short) 0xFF);
      ShortVector sumShort2 = acc0.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      subRet0 += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);

      sumShort1 = acc1.reinterpretAsShorts().and((short) 0xFF);
      sumShort2 = acc1.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      subRet1 += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);
    }
    // tail as bytes
    for (int r = limit; r < d.length; r++) {
      subRet0 += Integer.bitCount((q[r] & d[r]) & 0xFF);
      subRet1 += Integer.bitCount((q[r + d.length] & d[r]) & 0xFF);
    }
    ret += subRet0;
    ret += subRet1 << 1;
    return ret;
  }

  public static long ipByteBinBytePan(byte[] q, byte[] d) {
    if (VECTOR_BITSIZE < 256) {
      return ipByteBinBytePan128(q, d);
    }
    return ipByteBinPath512(q, d);
  }

  public static long ipByteBinPath512(byte[] q, byte[] d) {
    long ret = 0;
    long subRet0 = 0;
    long subRet1 = 0;
    long subRet2 = 0;
    long subRet3 = 0;
    int limit = ByteVector.SPECIES_PREFERRED.loopBound(d.length);
    int r = 0;
    LongVector sum0 = LongVector.zero(LongVector.SPECIES_PREFERRED);
    LongVector sum1 = LongVector.zero(LongVector.SPECIES_PREFERRED);
    LongVector sum2 = LongVector.zero(LongVector.SPECIES_PREFERRED);
    LongVector sum3 = LongVector.zero(LongVector.SPECIES_PREFERRED);

    for (; r < limit; r += ByteVector.SPECIES_PREFERRED.length()) {
      LongVector vq0 = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, q, r).reinterpretAsLongs();
      LongVector vq1 = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, q, r + d.length).reinterpretAsLongs();
      LongVector vq2 = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, q, r + d.length * 2).reinterpretAsLongs();
      LongVector vq3 = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, q, r + d.length * 3).reinterpretAsLongs();
      LongVector vd = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, d, r).reinterpretAsLongs();
      LongVector vres0 = vq0.and(vd).lanewise(VectorOperators.BIT_COUNT);
      LongVector vres1 = vq1.and(vd).lanewise(VectorOperators.BIT_COUNT);
      LongVector vres2 = vq2.and(vd).lanewise(VectorOperators.BIT_COUNT);
      LongVector vres3 = vq3.and(vd).lanewise(VectorOperators.BIT_COUNT);
      sum0 = sum0.add(vres0);
      sum1 = sum1.add(vres1);
      sum2 = sum2.add(vres2);
      sum3 = sum3.add(vres3);
    }
    subRet0 += sum0.reduceLanes(VectorOperators.ADD);
    subRet1 += sum1.reduceLanes(VectorOperators.ADD);
    subRet2 += sum2.reduceLanes(VectorOperators.ADD);
    subRet3 += sum3.reduceLanes(VectorOperators.ADD);
    // tail as ints
    for (; r < d.length-Integer.BYTES; r+=Integer.BYTES) {
      subRet0 += Integer.bitCount((int) BitUtil.VH_NATIVE_INT.get(q, r) & (int) BitUtil.VH_NATIVE_INT.get(d, r));
      subRet1 += Integer.bitCount((int) BitUtil.VH_NATIVE_INT.get(q, r + d.length) & (int) BitUtil.VH_NATIVE_INT.get(d, r));
      subRet2 += Integer.bitCount((int) BitUtil.VH_NATIVE_INT.get(q, r + 2 * d.length) & (int) BitUtil.VH_NATIVE_INT.get(d, r));
      subRet3 += Integer.bitCount((int) BitUtil.VH_NATIVE_INT.get(q, r + 3 * d.length) & (int) BitUtil.VH_NATIVE_INT.get(d, r));
    }
    // tail as bytes
    for (; r < d.length; r++) {
      subRet0 += Integer.bitCount((q[r] & d[r]) & 0xFF);
      subRet1 += Integer.bitCount((q[r + d.length] & d[r]) & 0xFF);
      subRet2 += Integer.bitCount((q[r + 2 * d.length] & d[r]) & 0xFF);
      subRet3 += Integer.bitCount((q[r + 3 * d.length] & d[r]) & 0xFF);
    }
    ret += subRet0;
    ret += subRet1 << 1;
    ret += subRet2 << 2;
    ret += subRet3 << 3;
    return ret;
  }

  public static long ipByteBinBytePan128(byte[] q, byte[] d) {
    long ret = 0;
    long subRet0 = 0;
    long subRet1 = 0;
    long subRet2 = 0;
    long subRet3 = 0;

    // TODO This is best for ARM, need to test on 256 & 512
    final int limit = BYTE_128_SPECIES.loopBound(d.length);
    // iterate in chunks of 256 bytes to ensure we don't overflow the accumulator
    // (256bytes/16lanes=16itrs)
    for (int j = 0; j < limit; j += 256) {
      ByteVector acc0 = ByteVector.zero(BYTE_128_SPECIES);
      ByteVector acc1 = ByteVector.zero(BYTE_128_SPECIES);
      ByteVector acc2 = ByteVector.zero(BYTE_128_SPECIES);
      ByteVector acc3 = ByteVector.zero(BYTE_128_SPECIES);
      int innerLimit = Math.min(limit - j, 256);
      for (int k = 0; k < innerLimit; k += BYTE_128_SPECIES.length()) {
        ByteVector vd = ByteVector.fromArray(BYTE_128_SPECIES, d, j + k);
        ByteVector vq0 = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k);
        ByteVector vq1 = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k + d.length);
        ByteVector vq2 = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k + 2 * d.length);
        ByteVector vq3 = ByteVector.fromArray(BYTE_128_SPECIES, q, j + k + 3 * d.length);
        ByteVector vres0 = vq0.and(vd);
        ByteVector vres1 = vq1.and(vd);
        ByteVector vres2 = vq2.and(vd);
        ByteVector vres3 = vq3.and(vd);
        vres0 = vres0.lanewise(VectorOperators.BIT_COUNT);
        vres1 = vres1.lanewise(VectorOperators.BIT_COUNT);
        vres2 = vres2.lanewise(VectorOperators.BIT_COUNT);
        vres3 = vres3.lanewise(VectorOperators.BIT_COUNT);
        acc0 = acc0.add(vres0);
        acc1 = acc1.add(vres1);
        acc2 = acc2.add(vres2);
        acc3 = acc3.add(vres3);
      }
      ShortVector sumShort1 = acc0.reinterpretAsShorts().and((short) 0xFF);
      ShortVector sumShort2 = acc0.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      subRet0 += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);

      sumShort1 = acc1.reinterpretAsShorts().and((short) 0xFF);
      sumShort2 = acc1.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      subRet1 += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);

      sumShort1 = acc2.reinterpretAsShorts().and((short) 0xFF);
      sumShort2 = acc2.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      subRet2 += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);

      sumShort1 = acc3.reinterpretAsShorts().and((short) 0xFF);
      sumShort2 = acc3.reinterpretAsShorts().lanewise(VectorOperators.LSHR, 8);
      subRet3 += sumShort1.add(sumShort2).reduceLanes(VectorOperators.ADD);
    }
    // tail as bytes
    for (int r = limit; r < d.length; r++) {
      subRet0 += Integer.bitCount((q[r] & d[r]) & 0xFF);
      subRet1 += Integer.bitCount((q[r + d.length] & d[r]) & 0xFF);
      subRet2 += Integer.bitCount((q[r + 2 * d.length] & d[r]) & 0xFF);
      subRet3 += Integer.bitCount((q[r + 3 * d.length] & d[r]) & 0xFF);
    }
    ret += subRet0;
    ret += subRet1 << 1;
    ret += subRet2 << 2;
    ret += subRet3 << 3;
    return ret;
  }

  private static final VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_128;
  private static final int VECTOR_SIZE = 16;
  private static final byte BYTE_MASK = (1 << B_QUERY) - 1;

  public static byte[] transposeBinPan(byte[] q, int D) {
    assert B_QUERY > 0;
    int B = (D + 63) / 64 * 64;
    byte[] quantQueryByte = new byte[B_QUERY * B / 8];
    int qOffset = 0;

    final byte[] v = new byte[32];
    final byte[] v1b = new byte[4];
    for (int i = 0; i < B; i += 32) {
      ByteVector q0 = ByteVector.fromArray(SPECIES, q, qOffset);
      ByteVector q1 = ByteVector.fromArray(SPECIES, q, qOffset + 16);

      ByteVector v0 = q0.lanewise(VectorOperators.LSHL, 8 - B_QUERY);
      ByteVector v1 = q1.lanewise(VectorOperators.LSHL, 8 - B_QUERY);
      v0 =
          v0.lanewise(
              VectorOperators.OR, q0.lanewise(VectorOperators.LSHR, B_QUERY).and(BYTE_MASK));
      v1 =
          v1.lanewise(
              VectorOperators.OR, q1.lanewise(VectorOperators.LSHR, B_QUERY).and(BYTE_MASK));

      for (int j = 0; j < B_QUERY; j++) {
        v0.intoArray(v, 0);
        v1.intoArray(v, 16);
        moveMaskEpi8Byte(v, v1b);
        for (int k = 0; k < 4; k++) {
          quantQueryByte[(B_QUERY - j - 1) * (B / 8) + i / 8 + k] = v1b[k];
          v1b[k] = 0;
        }

        v0 = v0.lanewise(VectorOperators.ADD, v0);
        v1 = v1.lanewise(VectorOperators.ADD, v1);
      }
      qOffset += 32;
    }
    return quantQueryByte;
  }

  public static byte[] transposeBinQ2(byte[] q, int D) {
    int B = (D + 63) / 64 * 64;
    byte[] quantQueryByte = new byte[2 * B / 8];
    int byte_mask = 1;
    for (int i = 0; i < 2 - 1; i++) {
      byte_mask = byte_mask << 1 | 0b00000001;
    }
    int qOffset = 0;
    final byte[] v1 = new byte[4];
    final byte[] v = new byte[32];
    for (int i = 0; i < B; i += 32) {
      // for every four bytes we shift left (with remainder across those bytes)
      int shift = 8 - 2;
      for (int j = 0; j < v.length; j += 4) {
        v[j] = (byte) (q[qOffset + j] << shift | ((q[qOffset + j] >>> (8 - shift)) & byte_mask));
        v[j + 1] =
            (byte)
                (q[qOffset + j + 1] << shift | ((q[qOffset + j + 1] >>> (8 - shift)) & byte_mask));
        v[j + 2] =
            (byte)
                (q[qOffset + j + 2] << shift | ((q[qOffset + j + 2] >>> (8 - shift)) & byte_mask));
        v[j + 3] =
            (byte)
                (q[qOffset + j + 3] << shift | ((q[qOffset + j + 3] >>> (8 - shift)) & byte_mask));
      }
      for (int j = 0; j < 2; j++) {
        moveMaskEpi8Byte(v, v1);
        for (int k = 0; k < 4; k++) {
          quantQueryByte[(2 - j - 1) * (B / 8) + i / 8 + k] = v1[k];
          v1[k] = 0;
        }
        for (int k = 0; k < v.length; k += 4) {
          v[k] = (byte) (v[k] + v[k]);
          v[k + 1] = (byte) (v[k + 1] + v[k + 1]);
          v[k + 2] = (byte) (v[k + 2] + v[k + 2]);
          v[k + 3] = (byte) (v[k + 3] + v[k + 3]);
        }
      }
      qOffset += 32;
    }
    return quantQueryByte;
  }

  public static byte[] transposeBin(byte[] q, int D) {
    assert B_QUERY > 0;
    int B = (D + 63) / 64 * 64;
    byte[] quantQueryByte = new byte[B_QUERY * B / 8];
    int byte_mask = 1;
    for (int i = 0; i < B_QUERY - 1; i++) {
      byte_mask = byte_mask << 1 | 0b00000001;
    }
    int qOffset = 0;
    final byte[] v1 = new byte[4];
    final byte[] v = new byte[32];
    for (int i = 0; i < B; i += 32) {
      // for every four bytes we shift left (with remainder across those bytes)
      int shift = 8 - B_QUERY;
      for (int j = 0; j < v.length; j += 4) {
        v[j] = (byte) (q[qOffset + j] << shift | ((q[qOffset + j] >>> (8 - shift)) & byte_mask));
        v[j + 1] =
            (byte)
                (q[qOffset + j + 1] << shift | ((q[qOffset + j + 1] >>> (8 - shift)) & byte_mask));
        v[j + 2] =
            (byte)
                (q[qOffset + j + 2] << shift | ((q[qOffset + j + 2] >>> (8 - shift)) & byte_mask));
        v[j + 3] =
            (byte)
                (q[qOffset + j + 3] << shift | ((q[qOffset + j + 3] >>> (8 - shift)) & byte_mask));
      }
      for (int j = 0; j < B_QUERY; j++) {
        moveMaskEpi8Byte(v, v1);
        for (int k = 0; k < 4; k++) {
          quantQueryByte[(B_QUERY - j - 1) * (B / 8) + i / 8 + k] = v1[k];
          v1[k] = 0;
        }
        for (int k = 0; k < v.length; k += 4) {
          v[k] = (byte) (v[k] + v[k]);
          v[k + 1] = (byte) (v[k + 1] + v[k + 1]);
          v[k + 2] = (byte) (v[k + 2] + v[k + 2]);
          v[k + 3] = (byte) (v[k + 3] + v[k + 3]);
        }
      }
      qOffset += 32;
    }
    return quantQueryByte;
  }

  private static void moveMaskEpi8Byte(byte[] v, byte[] v1b) {
    int m = 0;
    for (int k = 0; k < v.length; k++) {
      if ((v[k] & 0b10000000) == 0b10000000) {
        v1b[m] |= 0b00000001;
      }
      if (k % 8 == 7) {
        m++;
      } else {
        v1b[m] <<= 1;
      }
    }
  }

  private static long moveMaskEpi8(byte[] v) {
    long v1 = 0;
    for (int k = 0; k < v.length; k++) {
      if ((v[k] & 0b10000000) == 0b10000000) {
        v1 |= 0b00000001;
      } else {
        v1 |= 0b00000000;
      }
      if (k != v.length - 1) {
        v1 <<= 1;
      }
    }

    return v1;
  }

  public static float[] range(float[] q, float[] c) {
    float vl = Float.POSITIVE_INFINITY;
    float vr = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < q.length; i++) {
      float tmp = q[i] - c[i];
      if (tmp < vl) {
        vl = tmp;
      }
      if (tmp > vr) {
        vr = tmp;
      }
    }

    return new float[] {vl, vr};
  }

  public static QuantResult quantize(float[] q, float[] c, float[] u, float vl, float width) {
    // FIXME: speed up with panama?
    byte[] result = new byte[q.length];
    float oneOverWidth = 1.0f / width;
    int sumQ = 0;
    for (int i = 0; i < q.length; i++) {
      byte res = (byte) (((q[i] - c[i]) - vl) * oneOverWidth + u[i]);
      result[i] = res;
      sumQ += res;
    }

    return new QuantResult(result, sumQ);
  }
}
