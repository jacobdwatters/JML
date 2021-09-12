package com.jml.core.losses;

public interface LossFunction {

    /**
     * Function to measure the loss between expected values and actual values.
     *
     * @param expected Expected data values.
     * @param actual Actual data values
     * @return The loss between the expected and actual data values.
     */
    double compute(double[] expected, double[] actual);
}


