package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * An  interface for activation functions.
 */
public interface ActivationFunction {
    /**
     * Applies the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply activation function to.
     * @return The result of the element-wise activation function applied to the matrix.
     */
    public Matrix apply(Matrix data);


    /**
     * Applies the derivative of the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply the derivative of the activation function to.
     * @return
     */
    public Matrix slope(Matrix data);


    /**
     * Gets the name of the activation function.
     * @return The name of the activation function as a String.
     */
    public String getName();
}
