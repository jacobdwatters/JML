package com.jml.util;


public class ValueError {
    // Hide default constructor
    private ValueError() {throw new IllegalStateException("Can not instantiate utility class.");}


    /**
     * Checks if a value is non-negative
     * @return Returns true if the value is 0 or positive. Otherwise, returns false.
     */
    public static boolean isNonNegative(double val) {
        return val >= 0;
    }


    /**
     * Checks if a value is positive
     * @return Returns true if the value is  positive. Otherwise, returns false.
     */
    public static boolean isPositive(double val) {
        return val > 0;
    }
}
