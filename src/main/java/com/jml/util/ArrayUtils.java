package com.jml.util;

import java.lang.reflect.Array;
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
    public static String asString(int[] arr) {
        return asString(toObject(arr));
    }


    /**
     * Converts the array to a string.
     *
     * @param arr Array of interests.
     * @return The string representation of arr.
     */
    public static String asString(int[][] arr) {
        return asString(toObject(arr));
    }


    /**
     * Converts the array to a string.
     *
     * @param arr Array of interests.
     * @return The string representation of arr.
     */
    public static String asString(double[] arr) {
        return asString(toObject(arr));
    }


    /**
     * Converts the array to a string.
     *
     * @param arr Array of interests.
     * @return The string representation of arr.
     */
    public static String asString(double[][] arr) {
        return asString2D(toObject2D(arr));
    }


    /**
     * Converts the array to a string.
     *
     * @param arr Array of interests.
     * @return The string representation of arr.
     */
    public static String asString(Object[] arr) {
        StringBuilder arrAsString = new StringBuilder();

        for(int i=0; i<arr.length; i++) {
            arrAsString.append(arr[i].toString());

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
    public static String asString2D(Object[][] arr) {
        StringBuilder arrAsString = new StringBuilder();


        for(int i=0; i<arr.length; i++) {
            for(int j=0; j<arr[0].length; j++) {
                arrAsString.append(arr[i][j].toString());

                if(j!=arr[0].length-1) {
                    arrAsString.append(", ");
                }
            }

            if(i!=arr.length-1) {
                arrAsString.append(";\n");
            }
        }

        return arrAsString.toString();
    }


    /**
     * Converts an array to an Object array.
     * @return The array as an object array
     */
    public static Object[] toObject(Object val) {
        if (val instanceof Object[])
            return (Object[])val;
        int arrlength = Array.getLength(val);
        Object[] outputArray = new Object[arrlength];

        for(int i = 0; i < arrlength; ++i){
            outputArray[i] = Array.get(val, i);
        }

        return outputArray;
    }


    /**
     * Converts an array to an Object array.
     * @return The array as an object array
     */
    public static Object[][] toObject2D(Object val) {
        if (val instanceof Object[][])
            return (Object[][])val;
        int arrlength = Array.getLength(val);
        int arrlength2 = Array.getLength(Array.get(val, 0));
        Object[][] outputArray = new Object[arrlength][arrlength2];

        for(int i = 0; i < arrlength; ++i){
            for(int j=0; j<arrlength2; j++) {
                outputArray[i][j] = Array.get(Array.get(val, i), j);
            }
        }

        return outputArray;
    }


    /**
     * Converts a String array to an array of doubles.<br>
     * Also see {@link #toDouble(String[][])} for 2D arrays.
     *
     * @param arr Array to convert to doubles.
     * @return Content of arr in a double array.
     */
    public static double[] toDouble(String[] arr) {
        double[] arrAsDouble = new double[arr.length];

        for(int i=0; i<arr.length; i++) {
            arrAsDouble[i] = Double.valueOf(arr[i]);
        }

        return arrAsDouble;
    }


    /**
     * Converts a String array to an array of doubles.<br>
     * Also see {@link #toDouble(String[])} for 1D arrays.
     *
     * @param arr Array to convert to doubles.
     * @return Content of arr in a double array.
     */
    public static double[][] toDouble(String[][] arr) {
        double[][] arrAsDouble = new double[arr.length][arr[0].length];

        for(int i=0; i<arr.length; i++) {
            for(int j=0; j<arr[0].length; j++) {
                arrAsDouble[i][j] = Double.valueOf(arr[i][j]);
            }
        }

        return arrAsDouble;
    }
}
