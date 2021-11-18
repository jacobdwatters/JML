package com.jml.losses;

import linalg.Matrix;


/**
 * Functional Interface for a loss function. A loss function takes two parameters y and yPred. The loss function
 * is defined to be a function L such that L(y, yPred) >= 0.
 */
public interface Function {

    /**
     * Evaluates a loss function which takes features and targets for a model and weights as parameters.
     *
     * @param y True values.
     * @param yPred Expected values.
     * @return The loss between the expected and actual data values.
     */
    Matrix compute(Matrix y, Matrix yPred);
}


