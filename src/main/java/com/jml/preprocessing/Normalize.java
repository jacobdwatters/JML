package com.jml.preprocessing;

import com.jml.core.Stats;
import linalg.Matrix;
import linalg.Vector;

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
            normalization[i] = (data[i] - min)*(b-a)/(max-min); // Formula for min-max scaling in [a, b]
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
     * Normalizes the data by subtracting the meanNormalize and dividing by the L2-norm.
     *
     * @param data - data to normalize.
     * @return The L2-normalized data.
     */
    public static double[] l2(double[] data) { // TODO: Do we need to subtract mean here? sk-learn does not
        Matrix x = new Vector(data, 1);
        double mean = Stats.mean(data);
        Matrix m = new Matrix(1, data.length, mean);

        return x.sub(m).scalDiv(x.norm()).getValuesAsDouble()[0];
//        return x.scalDiv(x.norm()).getValuesAsDouble()[0];
    }


    /**
     * Normalizes each column of the data by subtracting the meanNormalize of that column and dividing by the
     * L2-norm of that column.
     *
     * @param data - data to normalize.
     * @return The L2-normalized data.
     */
    public static double[][] l2(double[][] data) {
        Matrix A = new Matrix(data).T();

        for(int i=0; i< data.length; i++) {
            A.setCol(l2(data[i]), i);
        }

        return A.T().getValuesAsDouble();
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
}
