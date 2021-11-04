package com.jml.neural_network.activations;

import linalg.Matrix;

public class Activations {

    private Activations() {
        throw new IllegalStateException("Cannot instantiate utility class.");
    }

    /**
     * Applies the sigmoid activation function, element-wise, to the data.<br>
     * Specifically, the logistic function is applied. i.e. 1/(1+e<sup>-x</sup>)
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    public static Activation sigmoid = (Matrix data) -> {
        Matrix result = new Matrix(data.shape());
        double value;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                value = 1 / (1+Math.exp(-data.get(i, j).re));
                result.set(value, i, j);
            }
        }

        return result;
    };
}
