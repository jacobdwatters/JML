package com.jml.neural_network.activations;


/**
 * A class which contains various activation functions.
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

    public static final ActivationFunction linear = new Linear();
}
