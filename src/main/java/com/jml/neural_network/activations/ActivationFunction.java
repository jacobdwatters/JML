package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * An  interface for activation functions.
 */
public interface ActivationFunction {

    /**
     * Applies the forward pass of activation function to a matrix.
     *
     * @param data The matrix to apply activation function to.
     * @return The result of the element-wise activation function applied to the matrix.
     */
    Matrix forward(Matrix data);


    /**
     * Applies backward pass of the activation function to a matrix (i.e. the derivative).
     *
     * @param data The matrix to apply the derivative of the activation function with respect to weights.
     * @return The backward pass of the activation function for the data matrix.
     */
    Matrix back(Matrix data);


    /**
     * Gets the name of the activation function.
     * @return The name of the activation function as a String.
     */
    String getName();
}
