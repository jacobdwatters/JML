package com.jml.neural_network.layers;

import com.jml.neural_network.activations.ActivationFunction;
import linalg.Matrix;


/**
 * An interface for specifying a layer for a neural network. Pre-made layers are available.<br><br>
 *
 * Pre-made Layers:
 * <pre>
 *     {@link com.jml.neural_network.layers.Dense Dense}
 *     {@link com.jml.neural_network.layers.Dropout Dropout}
 * </pre>
 */
public interface Layer {
    /**
     * Feeds the inputs through the layer.
     *
     * @param inputs Input values for the layer
     * @return The result of the layer applied to the values.
     */
    Matrix forward(Matrix inputs);


    /**
     * Computes the backward pass of the layer.
     *
     * @param previousVals Values of the previous layer in the network (or input values for first layer).
     * @param error Error for this layer.
     * @return Updates for weights based on the gradients of this layer.
     */
    Matrix[] back(Matrix previousVals, Matrix error);


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


    /**
     * Gets the weights of this layer in its matrix representation.
     * @return The weights of this layer.
     */
    Matrix getWeights();


    /**
     * Sets the weights for this layer.
     * @param w New weights for this layer
     */
    void setWeights(Matrix w);


    /**
     * Gets the matrix representation of the bias terms for this layer.
     * @return The bias terms of this layer.
     */
    Matrix getBias();


    /**
     * Sets the bias terms for this layer.
     * @param b New bias terms for this layer.
     */
    void setBias(Matrix b);


    /**
     * Gets the node values for this layer.
     * @return The node values of this layer in a matrix.
     */
    Matrix getValues();


    /**
     * Gets the details of this layer as a string.
     * @return A string representing the details of this layer. This may include input dimension, output dimension, and
     * activation function depending on the layer.
     */
    String inspect();


    /**
     * Constructs a string containing this layers activation function, input/output dimension,
     * and all parameters of the layer. Note, this will be more detailed than the getDetails() method and
     * can result in large strings.
     *
     * @return A string containing all information, including trainable parameters needed to recreate the layer.
     */
    String getDetails();


    /**
     * Gets the activation function for this layer.
     * @return The activation function for this layer. Note, if the layer does not have an activation function this will
     * be null.
     */
    ActivationFunction getActivation();
}
