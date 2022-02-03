package com.jml.neural_network.activations;

/**
 * A class which contains methods for geting various pre-made {@link com.jml.neural_network.activations.ActivationFunction activation functions} for use in neural network
 * {@link com.jml.neural_network.layers.BaseLayer layers}.
 */
public abstract class Activations {

    /**
     * The sigmoid activation function. f(x)=1/(1+exp(-x))
     */
    public static final ActivationFunction sigmoid = new Sigmoid();


    /**
     * Creates and returns a new instance of the sigmoid activation function.
     * @return The sigmoid activation function. f(x)=1/(1+exp(-x))
     */
    public static ActivationFunction sigmoid() {
        return new Sigmoid();
    }


    /**
     * A pre-defined instance of the Relu (Rectified Linear Unit) activation function. f(x)=max(0, x)
     */
    public static final ActivationFunction relu = new Relu();


    /**
     * Creates and returns a new relu activatoin function.
     * @return An instance of the Relu (Rectified Linear Unit) activation function. f(x)=max(0, x)
     */
    public static ActivationFunction relu() {
        return new Relu();
    }


    /**
     * A pre-defined instance of the linear activation function. f(x)=x.
     */
    public static final ActivationFunction linear = new Linear();


    /**
     * Creates and returns a new instacne of the linear activation function.
     * @return An instance of the linear activation function. f(x)=x.
     */
    public static ActivationFunction linear() {
        return new Linear();
    }


    /**
     * A pre-defined instance of the hyperbolic tangent activation function.
     * <code>f(x) = tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</code>
     */
    public static final ActivationFunction tanh = new Tanh();


    /**
     * Creates and returns a new instance of the hyperbolic tangent activation function.
     * @return An instance of the he hyperbolic tangent activation function<br>
     *      <code>f(x) = tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</code>
     */
    public static ActivationFunction tanh() {
        return new Tanh();
    }


    /**
     * Creates and returns a new instance of the softmax activations function.
     * @return
     *      The softmax activation function. f(<b>x</b>)<sub>i</sub> = exp(<b>x</b>_i) / sum<sub>j=1</sub><sup>m</sup>( exp(<b>x</b><sub>j</sub>) ) where
     *      <b>x</b> is a vector of length m and sum<sub>j=1</sub><sup>m</sup> ( exp(<b>x</b><sub>j</sub>) ) = exp(<b>x</b><sub>1</sub>) + exp(<b>x</b><sub>2</sub>) + ... + exp(<b>x</b><sub>m</sub>)
     */
    public static ActivationFunction softmax() {
        return new Softmax();
    }
}
