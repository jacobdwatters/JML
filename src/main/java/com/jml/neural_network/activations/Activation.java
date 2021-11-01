package com.jml.neural_network.activations;

import linalg.Matrix;


public abstract class Activation {

    // Applies the activation function to a data sample.
    public abstract double apply(double data);

    // Applies the activation function, element-wise, to the data.
    public abstract double[] apply(double[] data);

    // Applies the activation function, element-wise, to the data.
    public abstract double[][] apply(double[][] data);

    // Applies the activation function, element-wise, to the data.
    public abstract Matrix apply(Matrix data);
}
