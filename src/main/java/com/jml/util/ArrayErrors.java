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
     * @param arr Array in question.
     */
    public static void checkNotEmpty(double[] arr) {
        if(arr.length==0) {
            throw new IllegalArgumentException("Array is empty.");
        }
    }


    /**
     * Checks to see if two arrays have the same number of entries
     *
     * @throws IllegalArgumentException If arrays differ in length.
     * @param arr1 Array one in question.
     * @param arr2 Array two in question.
     */
    public static void checkSameLength(double[] arr1, double[] arr2) {
        if(arr1.length!=arr2.length) {
            throw new IllegalArgumentException("Arrays must have the same length but got " + arr1.length + " and " +
                    arr2.length);
        }
    }
}
