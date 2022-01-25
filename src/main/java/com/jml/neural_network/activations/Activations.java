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


    /**
     * The hyperbolic tangent activation function.
     * <code>f(x) = tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</code>
     */
    public static final ActivationFunction tanh = new Tanh();


    /**
     * The softmax activation function. f(<b>x</b>)<sub>i</sub> = exp(<b>x</b>_i) / sum<sub>j=1</sub><sup>m</sup>( exp(<b>x</b><sub>j</sub>) ) where
     * <b>x</b> is a vector of length m and sum<sub>j=1</sub><sup>m</sup> ( exp(<b>x</b><sub>j</sub>) ) = exp(<b>x</b><sub>1</sub>) + exp(<b>x</b><sub>2</sub>) + ... + exp(<b>x</b><sub>m</sub>).
     */
    public static final ActivationFunction softmax = new Softmax();
}
