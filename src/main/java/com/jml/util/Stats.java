package com.jml.util;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;

public class Stats {

    private Stats() {
        throw new IllegalStateException("Utility Class.");
    }


    public static double round(double value, int decimals) {
        double result;
        BigDecimal bd = BigDecimal.valueOf(value).setScale(decimals, RoundingMode.HALF_UP);;
        result = bd.doubleValue();
        return result;
    }


    /**
     * Computes the arithmetic mean.
     *
     * @param data Dataset to compute mean of.
     * @return arithmetic mean of the dataset.
     */
    public static double mean(double[] data) {
        ArrayErrors.checkNotEmpty(data); // Ensure the array is not empty
        double mean = 0;

        for(double point : data) {
            mean += point;
        }

        return mean / data.length;
    }


    /**
     * Computes the median.
     *
     * @param data Dataset to compute median of.
     * @return The median of the dataset.
     */
    public static double median(double[] data) {
        ArrayErrors.checkNotEmpty(data); // Ensure the array is not empty
        double median;

        if(data.length==1) {
            median = data[0]; // Then we have the median
        } else if(data.length==2) {
            median = (data[0] + data[1]) / 2; // No need to sort here
        } else {
            double[] sorted = new double[data.length];
            System.arraycopy(data, 0, sorted, 0, data.length);
            Arrays.sort(sorted);
            int index = (int) (data.length / 2);

            if (data.length % 2 == 0) { // Then we have an even number of entries.
                median = (sorted[index] + sorted[index - 1]) / 2;
            } else {
                median = sorted[index];
            }
        }

        return median;
    }


    /**
     * Computes the r<sup>2</sup> value or correlation between two sets of data.
     *
     * @param y Dataset one.
     * @param y_pred Dataset two.
     * @return The correlation coefficient for the given datasets.
     */
    public static double determination(double[] y, double[] y_pred) {
        ArrayErrors.checkSameLength(y, y_pred); // Ensure the arrays are the same length.

        double sst = sst(y);
        if(sst == 0) {
            throw new ArithmeticException("Division by zero will occur because sst=0.");
        }

        return 1-(sse(y, y_pred)/sst(y));
    }


    /**
     * Computes the r value or determination between two sets of data.
     *
     * @param y Dataset one.
     * @param y_pred Dataset two.
     * @return The coefficient of determination for the given datasets.
     */
    public static double correlation(double[] y, double[] y_pred) {
        return Math.sqrt(determination(y, y_pred));
    }


    /**
     * Computes the sum of square differences between two datasets.
     *
     * @param y Dataset one.
     * @param y_pred Dataset two.
     * @return The sum of square differences between two datasets.
     */
    public static double sse(double[] y, double[] y_pred) {
        double result = 0;

        ArrayErrors.checkSameLength(y, y_pred);

        for(int i=0; i<y.length; i++) {
            result += Math.pow(y[i]-y_pred[i], 2);
        }

        return result;
    }


    /**
     * Computes the sum of squares total of a dataset.
     *
     * @param y Dataset in question.
     * @return The sum of squares total.
     */
    public static double sst(double[] y) {
        double mean = mean(y),
                result = 0;

        for(int i=0; i<y.length; i++) {
            result += Math.pow(y[i]-mean, 2);
        }

        return result;
    }
}
