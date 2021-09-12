package com.jml.core.losses;

public class Loss {
    static LossFunction mse = (double[] actual, double[] expected) -> {
        if(actual.length != expected.length) {
            throw new IllegalArgumentException("actual and expected must be the same length but got " +
                    actual.length + " and " + expected.length);
        }
        double loss = 0;

        for(int i=0; i<actual.length; i++) {
            loss += Math.pow(actual[i]-expected[i], 2);
        }

        return loss/actual.length;
    };
}
