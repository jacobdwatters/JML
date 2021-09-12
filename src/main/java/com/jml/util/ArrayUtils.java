package com.jml.util;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class ArrayUtils {

    /**
     * Rounds all numbers in array to n decimal places.
     *
     * @param arr Array to round.
     * @param n Number of digits to round to.
     * @return Array with rounded numbers.
     */
    public static double[] round(double[] arr, int n) {
        double[] result = new double[arr.length];
        BigDecimal bd;

        for(int i=0; i<arr.length; i++) {
            bd = new BigDecimal(arr[i]).setScale(n, RoundingMode.HALF_UP);
            result[i] = bd.doubleValue();
        }

        return result;
    }


    public static String asString(double[] arr) {
        String arrAsString = "[";

        for(int i=0; i<arr.length; i++) {
            arrAsString += arr[i];

            if(i!=arr.length-1) {
                arrAsString += ", ";
            }
        }

        return arrAsString + "]";
    }

}
