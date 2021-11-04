package com.jml.preprocessing;

import com.jml.core.Stats;
import linalg.Matrix;
import linalg.Vector;


/**
 * Contains methods for normalizing data. These include
 * <pre>
 *     - {@link #minMaxScale(double[]) min-max scaling in (0, 1)}
 *     - {@link #minMaxScale(double[], double, double) min-max scaling in (a, b)}
 *     - {@link #meanNormalize(double[]) mean normalization}
 *     - {@link #l2(double[])  l2 normalization}
 *     - {@link #l1(double[])  l1 normalization}
 *     - {@link #zScore(double[]) Z-score normalization}
 * </pre>
 */
public class Normalize {

    // Private constructor to hide implicate one.
    private Normalize() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * Applies min-max feature scaling to data. This will rescale the data to be in [1, 0].<br><br>
     *
     * Also see {@link #minMaxScale(double[], double, double)}.
     *
     * @param data Dataset to apply normalization to.
     * @return A copy of the dataset that has been normalized using
     * min-max feature scaling.
     */
    public static double[] minMaxScale(double[] data) {
        return minMaxScale(data, 0, 1);
    }


    /**
     * Applies min-max feature scaling to data. This will rescale the data to be in [a, b].<br><br>
     *
     * Also see {@link #minMaxScale(double[])}.
     *
     * @param data Dataset to apply normalization to.
     * @param a Minimum value of the rescaled dataset.
     * @param b Maximum value of the rescaled dataset.
     * @return A copy of the dataset that has been normalized using
     * min-max feature scaling.
     */
    public static double[] minMaxScale(double[] data, double a, double b) {
        if(a >= b) {
            throw new IllegalArgumentException("a must be greater than b but got a=" + a  + " and b=" + b);
        }

        double[] normalization = new double[data.length];
        double min = Stats.min(data); // Extract the min value.
        double max = Stats.max(data); // Extract the max value.

        if(max==min) { // Then we can not apply this normalization.
            throw new ArithmeticException("max=min in the data which will cause division by zero.");
        }

        for(int i=0; i< data.length; i++) {
            normalization[i] = a + (data[i] - min)*(b-a)/(max-min); // Formula for min-max scaling in [a, b]
        }

        return normalization;
    }


    /**
     * Applies meanNormalize normalization to the data.
     *
     * @param data Dataset to apply meanNormalize normalization to.
     * @return A copy of the dataset which has been normalized using meanNormalize normalization.
     */
    public static double[] meanNormalize(double[] data) {

        double[] normalization = new double[data.length];
        double mean = Stats.mean(data);
        double min = Stats.min(data);
        double max = Stats.max(data);

        if(max==min) { // Then we can not apply this normalization.
            throw new ArithmeticException("max=min in the data which will cause division by zero.");
        }

        for(int i=0; i< data.length; i++) {
            normalization[i] = (data[i] - mean) / (max-min);
        }

        return normalization;
    }


    /**
     * Normalizes the data by subtracting the mean and dividing by the L1-norm.
     *
     * @param data - data to normalize.
     * @return The L1-normalized data.
     */
    public static double[] l1(double[] data) {
        Matrix x = new Vector(data, 1);
        double mean = Stats.mean(data);
        Matrix m = new Matrix(1, data.length, mean);

        return x.scalDiv(x.norm(1)).getValuesAsDouble()[0];
    }


    /**
     * Normalizes the data by subtracting the mean and dividing by the L2-norm.
     *
     * @param data - data to normalize.
     * @return The L2-normalized data.
     */
    public static double[] l2(double[] data) {
        Matrix x = new Vector(data, 1);
        double mean = Stats.mean(data);
        Matrix m = new Matrix(1, data.length, mean);

        return x.scalDiv(x.norm()).getValuesAsDouble()[0];
    }


    /**
     * Normalizes each column of the data by subtracting the mean of that column and dividing by the
     * L2-norm of that column.
     *
     * @param data - data to normalize.
     * @return The L2-normalized data.
     */
    public static double[][] l2(double[][] data) {
        double[][] normalization = new double[data.length][data[0].length];

        for(int i=0; i< data.length; i++) {
            normalization[i] = l2(data[i]);
        }

        return normalization;
    }


    /**
     * Applies Z-score normalization to the dataset.
     *
     * @param data The dataset of interest.
     * @return A copy of the dataset which has been normalized using Z-score normalization.
     */
    public static double[] zScore(double[] data) {

        double[] normalization = new double[data.length];
        double std = Stats.std(data);
        double mean = Stats.mean(data);

        for(int i=0; i<data.length; i++) {
            normalization[i] = (data[i]-mean) / std; // Apply the Z-score normalization to each entry.
        }

        return normalization;
    }


    /**
     * Applies Z-score normalization to the dataset.
     *
     * @param data The dataset of interest.
     * @return A copy of the dataset which has been normalized using Z-score normalization.
     */
    public static double[][] zScore(double[][] data) {

        // TODO: Should this normalize each feature? So the columns, not the rows.

        // TODO: In order to scale validation data, we need to know the mean and standard deviation of the training data.
        //  So there should be a Normalize object which will be "fit" to the data. Then the same scaling can be applied to the validation.
        double[][] normalization = new double[data.length][data[0].length];

        for(int i=0; i<data.length; i++) {
            normalization[i] = zScore(data[i]); // Apply the Z-score normalization to each entry.
        }

        return normalization;
    }
}
