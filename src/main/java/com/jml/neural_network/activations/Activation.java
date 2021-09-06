package com.jml.neural_network.activations;

public abstract class Activation<E> {

    /**
     * Applies the activation function to the data.
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    public abstract E  apply(E data);
}
