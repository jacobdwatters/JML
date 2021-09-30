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
            bd = BigDecimal.valueOf(arr[i]).setScale(n, RoundingMode.HALF_UP);
            result[i] = bd.doubleValue();
        }

        return result;
    }


    /**
     * Rounds all numbers in array to n decimal places.
     *
     * @param arr Array to round.
     * @param n Number of digits to round to.
     * @return Array with rounded numbers.
     */
    public static double[][] round(double[][] arr, int n) {
        double[][] result = new double[arr.length][arr[0].length];

        for(int i=0; i<arr.length; i++) {
            result[i] = round(arr[i], n);
        }

        return result;
    }


    /**
     * Converts the array to a string.
     *
     * @param arr Array of interests.
     * @return The string representation of arr.
     */
    public static String asString(double[] arr) {
        StringBuilder arrAsString = new StringBuilder("");

        for(int i=0; i<arr.length; i++) {
            arrAsString.append(arr[i]);

            if(i!=arr.length-1) {
                arrAsString.append(", ");
            }
        }

        return arrAsString.toString();
    }


    /**
     * Converts the array to a string.
     *
     * @param arr Array of interests.
     * @return The string representation of arr.
     */
    public static String asString(double[][] arr) {
        StringBuilder arrAsString = new StringBuilder("");


        for(int i=0; i<arr.length; i++) {
            for(int j=0; j<arr[0].length; j++) {
                arrAsString.append(arr[i][j]);

                if(j!=arr.length-1) {
                    arrAsString.append(", ");
                }
            }

            arrAsString.append("\n");
        }

        return arrAsString.toString();
    }

}
