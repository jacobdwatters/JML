package com.jml.neural_network.activations;

import linalg.Matrix;

public interface Activation {
    public abstract Matrix apply(Matrix data);
}
