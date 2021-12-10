package com.jml.core;

import java.util.Arrays;

public class Metrics {
    private Metrics() {
        throw new IllegalStateException("Cannot instantiate utility method.");
    }


    /**
     * Computes the accuracy between predictions and the true targets.
     *
     * @param predictions Predictions of the true targets.
     * @param targets True targets.
     * @return The percentage of correct predictions
     */
    public static double accuracy(double[][] predictions, double[][] targets) {
        double accuracy = 0;

        if(predictions.length != targets.length || predictions[0].length != targets[0].length) {
            throw new IllegalArgumentException("Predictions and targets must have the same shape but got (" +
                    predictions.length + ", " + predictions[0].length + ") and ("
                    + targets.length + ", " + targets[0].length + ").");
        }

        for(int i=0; i<predictions.length; i++) {
            if(Arrays.equals(predictions[i], targets[i])) {
                accuracy++;
            }
        }

        return accuracy/predictions.length;
    }


    /**
     * Computes the accuracy between predictions and the true targets.
     *
     * @param predictions Predictions of the true targets.
     * @param targets True targets.
     * @return The percentage of correct predictions
     */
    public static double accuracy(double[] predictions, double[] targets) {
        double accuracy = 0;

        if(predictions.length!= targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same number of samples but got " +
                    predictions.length + " and " + targets.length);
        }

        for(int i=0; i<predictions.length; i++) {
            if(predictions[i]==targets[i]) {
                accuracy++;
            }
        }

        return accuracy/predictions.length;
    }
}
