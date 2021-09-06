package com.jml.util;

import java.util.Arrays;
import com.jml.util.ArrayErrors;

public class Stats {


    /**
     * Computes the arithmetic mean.
     *
     * @param data Dataset to compute mean of.
     * @return arithmetic mean of the dataset.
     */
    public static double mean(double[] data) {
        ArrayErrors.notEmpty(data); // Ensure the array is not empty
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
        ArrayErrors.notEmpty(data); // Ensure the array is not empty
        double median = 0;

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
                median = (data[index] + data[index - 1]) / 2;
            } else {
                median = data[index];
            }
        }

        return median;
    }
}
