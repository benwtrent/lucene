package org.apache.lucene.sandbox.rabitq;

public record Result(float sqrY, int c, Long originalCDist) implements Comparable {

  public Result(float sqrY, int c) {
    this(sqrY, c, null);
  }

  @Override
  public int compareTo(Object obj) {
    if (obj instanceof Result) {
      if (Float.isNaN(((Result) obj).sqrY)) {
        return 1;
      } else {
        if (this.sqrY == ((Result) obj).sqrY) {
          return 0;
        } else if (this.sqrY < ((Result) obj).sqrY) {
          return -1;
        } else {
          return 1;
        }
      }
    } else {
      return 1;
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof Result) {
      return this.sqrY == ((Result) obj).sqrY && this.c == ((Result) obj).c;
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Float.hashCode(sqrY) * 31 + Integer.hashCode(c);
  }

  @Override
  public String toString() {
    return this.sqrY + ":" + this.c + (originalCDist != null ? ":" + originalCDist : "");
  }
}
