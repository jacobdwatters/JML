package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * A functional interface for activation functions. All activation functions should take and return a Matrix and be
 * implemented as lambdas.
 */
public interface Activation {
    public abstract Matrix apply(Matrix data);
}
