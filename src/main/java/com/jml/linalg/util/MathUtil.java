package com.jml.linalg.util;

import java.util.stream.LongStream;

public class MathUtil {
	public static long fact(int n) {
	    return LongStream.rangeClosed(1, n)
	        .reduce(1, (long x, long y) -> x * y);
	}
}
