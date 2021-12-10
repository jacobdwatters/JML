package com.jml.neural_network.activations;

/**
 * A class which contains various pre-made {@link com.jml.neural_network.activations.ActivationFunction activation functions} for use in neural network
 * {@link com.jml.neural_network.layers.Layer layers}.
 */
public abstract class Activations {

    /**
     * The sigmoid activation function. f(x)=1/(1+exp(-x))
     */
    public static final ActivationFunction sigmoid = new Sigmoid();


    /**
     * The Relu (Rectified Linear Unit) activation function. f(x)=max(0, x)
     */
    public static final ActivationFunction relu = new Relu();


    /**
     * The linear activation function. f(x)=x.
     */
    public static final ActivationFunction linear = new Linear();
}
