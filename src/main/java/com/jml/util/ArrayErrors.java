package com.jml.util;

public class ArrayErrors {

    /**
     * Checks to see if array is empty.
     *
     * @throws IllegalArgumentException If array is empty
     * @param arr
     */
    public static void notEmpty(double[] arr) {
        if(arr.length==0) {
            throw new IllegalArgumentException();
        }
    }
}
