package com.jml.neural_network.layers;

import linalg.Matrix;


/**
 * Base layer interface. Specifies basic functionality that all layers should have.
 */
public interface BaseLayer {


    /**
     * Computes forward pass for layer.
     * @return Result of the forward pass of a layer as a matrix.
     */
    Matrix forward(Matrix input);


    /**
     * Computes backward pass for layer.
     * @return Result of the backwards pass of the layer as a Matrix. If this layer has weights, this matrix
     * will have the same shape as the weight matrix for the layer.
     */
    Matrix back(Matrix upStream);


    /**
     * Gets the input dimension of this layer.
     * @return The input dimension of this layer.
     */
    int getInDim();


    /**
     * Gets the output dimension of this layer.
     * @return The output dimension of this layer.
     */
    int getOutDim();


    /**
     * Updates this layers input dimension. This is useful for creating a layer with an unknown input dimension and
     * inferring it from the previous layer in the network.
     * @param newInDim New input dimension for layer.
     */
    void updateInDim(int newInDim);


    /**
     * Gets the trainable parameters of this layer in an array.
     * @return The trainable parameters of this layer in an array. If this layer does not inherit
     * {@link TrainableLayer TrainableLayer} then this will return null.
     */
    default Matrix[] getParams() {
        return null;
    }


    /**
     * Sets parameters of this layer. If this layer does not inherit {@link TrainableLayer TrainableLayer}
     * then this does nothing.
     * @param params Parameters of layer as an array of matrices.
     */
    default void setParams(Matrix... params) {
        // Does nothing.
    }


    /**
     * Gets the update matrices for the trainable parameters of this layer.
     * @return An array of update matrices for this layers' trainable parameters. If this layer does not inherit
     *      * {@link TrainableLayer TrainableLayer} then this will return null.
     */
    default Matrix[] getUpdates() {
        return null;
    }


    /**
     * Resets the gradients for this layers' trainable parameters.
     */
    default void resetGradients() {
        // Does nothing.
    }


    /**
     * Gets and formats details of this layer in a human-readable String.
     * @return The details of this layer in a human-reusable String.
     */
    String inspect();


    /**
     * Constructs a string containing this all details of the model pertinent for saving the model to a file.
     *
     * @return A string containing all information, including trainable parameters, needed to recreate the layer.
     */
    String getDetails();
}
