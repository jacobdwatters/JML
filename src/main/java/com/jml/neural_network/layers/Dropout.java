package com.jml.neural_network.layers;

import com.jml.neural_network.activations.Activation;
import linalg.Matrix;


/**
 * A dropout layer. This layer has a probability of dropping (zeroing out) each element during the forward pass. This
 * is an effective form of regularization. In addition, the outputs of this layer are scaled by 1/(1-p) where p is the
 * probability of dropping an element of the layer.
 */
public class Dropout implements Layer {

    /**
     * Probability of element being zeroed.
     */
    double p;
    int inDim = -1; // Input size for the layer. The output size will be the same.

    /**
     * Constructs a dropout layer for a neural network.<br>
     * <b><u>Note</b></u>: this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Dropout(int, double)} to specify the input dimension.
     *
     * @param p The probability of an element being zeroed.
     */
    public Dropout(double p) {
        this.p = p;
    }


    /**
     * Constructs a dropout layer for a neural network.
     *
     * @param inDim Input dimension of the layer.
     * @param p The probability of an element being zeroed.
     */
    public Dropout(int inDim, double p) {
        this.inDim = inDim;
        this.p = p;
    }

    /**
     * Feeds the inputs through the layer.
     *
     * @param inputs Input values for the layer
     * @return The result of the layer applied to the values.
     */
    @Override
    public Matrix forward(Matrix inputs) {
        // TODO: AUto-generated method stub.
        return null;
    }


    @Override
    public int getInDim() {
        return this.inDim;
    }


    @Override
    public int getOutDim() {
        return this.inDim;
    }


    /**
     * Updates the input dimension for the layer. <br>
     * <b><u>WARNING:</u></b> This will zero any weight values the layer may currently be holding.
     *
     * @param inDim New input size for the layer.
     */
    @Override
    public void updateInDim(int inDim) {
        this.inDim = inDim;
    }
}
