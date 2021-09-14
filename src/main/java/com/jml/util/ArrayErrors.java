package com.jml.util;

public class ArrayErrors {


    // Private constructor to hide implicate one.
    private ArrayErrors() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }

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
