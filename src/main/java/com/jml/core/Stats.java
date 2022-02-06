package com.jml.core;

import com.jml.util.ArrayErrors;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;


/**
 * The stats class is a utility class to compute various statistical information about datasets.
 */
public class Stats {

    private static final SplittableRandom random = new SplittableRandom();

    private Stats() {
        throw new IllegalStateException("Utility Class.");
    }

    public static double round(double value, int decimals) {
        double result;
        BigDecimal bd = BigDecimal.valueOf(value).setScale(decimals, RoundingMode.HALF_UP);
        result = bd.doubleValue();
        return result;
    }


    /**
     * Computes the arithmetic mean.
     *
     * @param data Dataset to compute mean of.
     * @return arithmetic mean of the dataset.
     */
    public static double mean(double... data) {
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
    public static double median(double... data) {
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
            int index = data.length / 2;

            if (data.length % 2 == 0) { // Then we have an even number of entries.
                median = (sorted[index] + sorted[index - 1]) / 2;
            } else {
                median = sorted[index];
            }
        }

        return median;
    }


    /**
     * Computes the mode of a dataset.
     *
     * @param data Dataset to compute the mode of.
     * @return The mode of the dataset.
     */
    public static double mode(double... data) {
        double mode = 0;
        int maxCount = 0, i, j;

        for (i = 0; i < data.length; ++i) {
            int count = 0;
            for (j = 0; j < data.length; ++j) {
                if (data[j] == data[i])
                    ++count;
            }

            if (count > maxCount) {
                maxCount = count;
                mode = data[i];
            }
        }

       return mode;
    }


    /**
     * Computes the variance for the data set. This is similar to the mean squared error but the
     * {@link #sst(double[]) sst} is divided by (n-1) where n is the number of obervations in the dataset.
     *
     * @param data Dataset of interest.
     * @return The variance of the data.
     */
    public static double variance(double... data) {
        if(data.length < 2) {
            throw new IllegalArgumentException("Variance requires at least two data points.");
        }

        return sst(data)/(data.length-1);
    }


    /**
     * Computes the standard deviation of the dataset. It is assumed that data contains only a sample of
     * observations from the true population.
     *
     * @param data Dataset of interest
     * @return The standard deviation of the data.
     */
    public static double std(double... data) {
        return Math.sqrt(sst(data)/data.length);
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

        return 1-(sse(y, y_pred)/sst);
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
    public static double sst(double... y) {
        double mean = mean(y),
                result = 0;

        for(int i=0; i<y.length; i++) {
            result += Math.pow(y[i]-mean, 2);
        }

        return result;
    }


    /**
     * Finds the minimum value in a dataset.
     *
     * @param data Dataset of interest.
     * @return The minimum value in data.
     */
    public static double min(double... data) {
        double minimum = Double.MAX_VALUE;

        for(int i=0; i< data.length; i++) {
            if(data[i] < minimum) { // Then we have a new minimum
                minimum = data[i];
            }
        }

        return minimum;
    }


    /**
     * Finds index of minimum value in an array.
     *
     * @param data The array to find index of minimum.
     * @return The index of the entry with the smallest value.
     */
    public static int minIndex(double[] data) {
        double minimum = Double.MAX_VALUE;
        int mindex = -1;

        for(int i=0; i< data.length; i++) {
            if(data[i] < minimum) { // Then we have a new minimum
                minimum = data[i];
                mindex = i;
            }
        }

        return mindex;
    }



    /**
     * Finds indices of the k smallest values in an array.
     *
     * @param data The array to find indices of the smallest values.
     * @return An array of length k containing the indices of the k smallest values.
     */
    public static int[] minIndices(double[] data, int k) {
        if(k>data.length) {
            throw new IllegalArgumentException("k can not be greater than the length of the array but got k=" + k +
                    " and an array length of " + data.length);
        }

        int[] mindices = new int[k];

        Map<Integer, Double> hm = new HashMap<>();

        for(int i=0; i<data.length; i++) { // Fill hashmap
            hm.put(i, data[i]);
        }

        // --------------------------------------- SORT HASHMAP --------------------------------
        List<Map.Entry<Integer, Double> > list =
                new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet());

        // Sort the list
        list.sort(new Comparator<Map.Entry<Integer, Double>>() {
            public int compare(Map.Entry<Integer, Double> o1,
                               Map.Entry<Integer, Double> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<Integer, Double> sorted = new LinkedHashMap<Integer, Double>();
        for (Map.Entry<Integer, Double> aa : list) {
            sorted.put(aa.getKey(), aa.getValue());
        }
        // -------------------------------------------------------------------------------------

        Object[] keys = sorted.keySet().toArray();

        for(int i=0; i<k && i<keys.length; i++) {
            mindices[i] = (int) keys[i];
        }

        return mindices;
    }


    /**
     * Finds the maximum value in a dataset.
     *
     * @param data Dataset of interest.
     * @return The maximum value in data.
     */
    public static double max(double... data) {
        double maximum = -Double.MAX_VALUE;

        for(int i=0; i< data.length; i++) {
            if(data[i] > maximum) { // Then we have a new maximum
                maximum = data[i];
            }
        }

        return maximum;
    }


    /**
     * Computes the sum of an array of values.
     *
     * @param data Data to sum.
     * @return The sum of all entries of the data array.
     */
    public static double sum(double... data) {
        double sum=0;

        for(double value  : data) {
            sum+=value;
        }

        return sum;
    }


    /**
     * Generates a random boolean with a specified probability of being true.
     *
     * @param p Probability of being true. Must be in range [0, 1].
     * @return Returns a random boolean with probability <code>p</code> of being true.
     * @throws IllegalArgumentException if <code>p</code> is not in range [0, 1].
     */
    public static boolean genRandBoolean(double p) {
        if(p<0 || p>1) {
            throw new IllegalArgumentException("probability must be between 0 and 1 inclusive but got " + p + ".");
        }

        return random.nextDouble() < p;
    }
}
