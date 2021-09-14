package com.jml.core.losses;


/**
 * This class contains lambda functions for various loss functions including:
 * <pre>
 *     - mse: mean-squared-error.
 * </pre>
 */
public class Loss {

    // A private constructor to hide the implicit constructor.
    private Loss() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * The mean-squared-error loss function.<br>
     * That is <code>mse = (1/n)sum(x<sub>i</sub> - y<sub>i</sub>)</code>
     * where <code>x<code/> and <code>y<code/> are datasets of length <code>n<code/>,
     * and <code>x<code/> is the actual data and <code>y<code/> is the predicted data.
     */
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
