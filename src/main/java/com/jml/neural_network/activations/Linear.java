package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * The linear activation function. Applies the activation f(x)= x to the data. This is equivalent to having no activation
 * for a layer.
 */
public class Linear implements ActivationFunction {

    public final String NAME = "Linear";

    /**
     * Applies the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply activation function to.
     * @return The result of the element-wise activation function applied to the matrix.
     */
    @Override
    public Matrix apply(Matrix data) {
        return data.copy();
    }


    /**
     * Applies the derivative of the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply the derivative of the activation function to.
     * @return The slope of the activation function evaluated, element-wise, of the matrix.
     */
    @Override
    public Matrix slope(Matrix data) {
        return Matrix.ones(data.shape());
    }


    /**
     * Gets the name of the activation function.
     *
     * @return The name of the activation function as a String.
     */
    @Override
    public String getName() {
        return NAME;
    }
}
