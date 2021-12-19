package com.jml.util;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Random;

public class ArrayUtils {
    static Random rand = new Random();

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


    /**
     * Converts a String array to an 1D array of doubles.<br>
     * Also see {@link #toDouble(String[])} for 1D arrays.
     *
     * @param arr Array to convert to doubles.
     * @return Content of arr in a 1D double array.
     */
    public static double[] toDoubleFlat(String[][] arr) {
        double[] arrAsDouble = new double[arr.length*arr[0].length];
        int result_i = 0;

        for(int i=0; i<arr.length; i++) {
            for(int j=0; j<arr[0].length; j++) {
                arrAsDouble[result_i] = Double.valueOf(arr[i][j]);
                result_i++;
            }
        }

        return arrAsDouble;
    }


    /**
     * Converts a String array to an 1D array of doubles.<br>
     * Also see {@link #toDouble(String[])} for 1D arrays.
     *
     * @param arr Array to convert to doubles.
     * @return Content of arr in a 1D double array.
     */
    public static int[] toIntFlat(String[][] arr) {
        int[] arrAsInt = new int[arr.length*arr[0].length];
        int result_i = 0;

        for(int i=0; i<arr.length; i++) {
            for(int j=0; j<arr[0].length; j++) {
                arrAsInt[result_i] = Double.valueOf(arr[i][j]).intValue();
                result_i++;
            }
        }

        return arrAsInt;
    }


    /**
     * Converts a double array to an int array. Note, this is not the same as round(arr, 0) as this method does not
     * round the double value just casts it and any precision after the decimal place is lost.
     *
     * @param arr Array to convert to int array.
     * @return The given array converted to an int array.
     */
    public static int[] toInt(double[] arr) {
        int[] arrAsInt = new int[arr.length];

        for (int i=0; i<arr.length; ++i)
            arrAsInt[i] = (int) arr[i];

        return arrAsInt;
    }


    /**
     * Randomly shuffles arrays using the Fisher–Yates algorithm.
     *
     * @param arr Arrays to shuffle
     * @return Arrays with rows randomly shuffled.
     */
    public static double[] shuffle(double arr[]) {
        double[] newArr = arr.clone();
        double temp;

        for (int i = arr.length-1; i>0; i--) {

            // Pick a random index from 0 to i
            int j = rand.nextInt(i+1);

            // Swap arr[i] with the element at random index
            temp = newArr[i];
            newArr[i] = newArr[j];
            newArr[j] = temp;
        }

        return newArr;
    }


    /**
     * Randomly shuffles arrays using the Fisher–Yates algorithm.<br><br>
     *
     * If more than one array is passed, the shuffled indices of all arrays will be the same.
     * I.e. if the following arrays are passed:
     * <pre>
     *     arr1: [1, 2, 3, 4, 5]
     *     arr2: [10, 20, 30, 40, 50]
     *
     *     Then, if arr1 is shuffled to: [3, 1, 2, 5, 4]
     *     arr2 will be shuffled to: [30, 10, 20, 50, 40]
     * </pre>
     *
     *
     * @param arr Arrays to shuffle
     * @return Arrays with rows randomly shuffled.
     */
    public static double[][] shuffle(double[]... arr) {
        double[][] newArr = arr.clone();
        double temp;

        for (int i = arr[0].length-1; i>0; i--) {

            // Pick a random index from 0 to i
            int j = rand.nextInt(i+1);

            for(int k=0; k<arr.length; k++) { // Swap same element for all rows.
                // Swap arr[i] with the element at random index
                temp = newArr[k][i];
                newArr[k][i] = newArr[k][j];
                newArr[k][j] = temp;
            }
        }

        return newArr;
    }


    /**
     * Randomly shuffles 2D arrays by rows using the Fisher–Yates algorithm.<br><br>
     *
     * If more than one array is passed, the shuffled row indices of all arrays will be the same.
     * I.e. if the following arrays are passed:
     * <pre>
     *     arr1: [1, 2, 3],
     *           [4, 5, 6],
     *           [7, 8, 9]
     *
     *     arr2: [10, 20, 30],
     *           [40, 50, 60],
     *           [70, 80, 90]
     *
     *     Then, if arr1 is shuffled to: [4, 5, 6],
     *                                   [1, 2, 3],
     *                                   [7, 8, 9]
     *
     *     arr2 will be shuffled to: [40, 50, 60],
     *                               [10, 20, 30],
     *                               [70, 80, 90]
     * </pre>
     *
     *
     * @param arr 2D Arrays to shuffle
     * @return Arrays with rows randomly shuffled.
     */
    public static double[][][] shuffle(double[][]... arr) {
        double[][][] newArr = arr.clone();
        double[] temp;

        for (int i = arr[0].length-1; i>0; i--) {

            // Pick a random index from 0 to i
            int j = rand.nextInt(i+1);


            for(int k=0; k<arr.length; k++) { // Swap same row for entire depth.
                // Swap arr[i] with the element at random index
                temp = newArr[k][i];
                newArr[k][i] = newArr[k][j];
                newArr[k][j] = temp;
            }
        }

        return newArr;
    }
}
