package com.jml.neural_network.layers;

import linalg.Matrix;

public interface Layer {
    /**
     * Feeds the inputs through the layer.
     *
     * @param inputs Input values for the layer
     * @return The result of the layer applied to the values.
     */
    Matrix forward(Matrix inputs);

    /**
     * Updates the input dimension for the layer. <br>
     * <b><u>WARNING:</u></b> This will zero any weight values the layer may currently be holding.
     *
     * @param inDim New input size for the layer.
     */
    void updateInDim(int inDim);

    /**
     * Gets the input dimension for this layer.
     *
     * @return The input dimension of this layer.
     */
    int getInDim();

    /**
     * Gets the output dimension for this layer.
     *
     * @return The output dimension of this layer.
     */
    int getOutDim();

    Matrix getWeights();

    void setWeights(Matrix w);

    void setBias(Matrix b);

    /**
     * Gets the node values for this layer.
     * @return The node values of this layer in a matrix.
     */
    Matrix getValues();

    Matrix getBias();

    String getDetails();
}
