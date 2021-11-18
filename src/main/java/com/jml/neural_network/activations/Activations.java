package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * This class contains several activation functions for use in a neural network. The activation functions are lambdas
 * that implement the {@link com.jml.neural_network.activations.Activation} functional interface.
 */
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


    /**
     * Computes the derivative of the sigmoid function evaluated at each element of the data matrix.
     */
    public static Activation sigmoidSlope = (Matrix data) -> {
        Matrix result = new Matrix(data.shape());
        double value, exp;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                exp = Math.exp(-data.get(i, j).re);
                value = exp / Math.pow(1+exp, 2);
                result.set(value, i, j);
            }
        }

        return result;
    };


    /**
     * Applies a linear activation. When used on a layer for a neural network
     * this is equivalent to having no activation resulting in a linear layer.
     */
    public static Activation linear = (Matrix data) -> {
        return new Matrix(data);
    };
}
