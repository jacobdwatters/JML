package com.jml.core;

import com.jml.util.Stats;
import linalg.Matrix;
import linalg.Vector;

public class Normalize {


    // Private constructor to hide implicate one.
    private Normalize() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * Applies min-max feature scaling to data.
     *
     * @param data Dataset to apply normalization to.
     * @return A copy of the dataset that has been normalized using
     * min-max feature scaling.
     */
    public static double[] minMaxScale(double[] data) {
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * Normalizes the data by subtracting the mean and dividing by the L2-norm.
     *
     * @param data - data to normalize.
     * @return The L2-normalized data.
     */
    public static double[] l2Normalize(double[] data) {
        Matrix x = new Vector(data, 1);
        double mean = Stats.mean(data);
        Matrix m = new Matrix(1, data.length, mean);

        return x.sub(m).scalDiv(x.norm()).getValuesAsDouble()[0];
    }
}
