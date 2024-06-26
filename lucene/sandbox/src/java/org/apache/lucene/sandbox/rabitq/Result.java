package org.apache.lucene.sandbox.rabitq;

record Result(float sqrY, int c) implements Comparable {
    @Override
    public int compareTo(Object obj) {
        if(obj instanceof Result) {
            if ( this.sqrY == ((Result) obj).sqrY) {
                return 0;
            }
            else if(this.sqrY < ((Result) obj).sqrY) {
                return -1;
            } else {
                return 1;
            }
        } else {
            return 1;
        }
    }

    @Override
    public boolean equals(Object obj) {
        if(obj instanceof Result) {
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
        return this.sqrY + ":" + this.c;
    }
}
