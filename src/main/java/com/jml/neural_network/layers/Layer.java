package com.jml.neural_network.layers;

public abstract class Layer<E, F> {

    E inDim = null;
    F outDim = null;
    String activation = null;

    /**
     * Constructs a layer for a neural network.
     *
     * @param inDim Layer input dimension.
     * @param outDim Layer output dimension.
     * @param activation Activation function for layer.
     */
    public Layer(E inDim, F outDim, String activation) {
        this.inDim = inDim;
        this.outDim = outDim;
        this.activation = activation;
    }
}
