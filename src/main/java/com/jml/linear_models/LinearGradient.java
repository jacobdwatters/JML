package com.jml.linear_models;

import linalg.Matrix;

/**
 * A utility class for computing the gradient the SSE loss of a linear function.
 */
class LinearGradient {

    /**
     * Computes gradient of linear regression model.
     * @param X Input matrix for model.
     * @param y Output matrix for model.
     * @param w Parameter matrix for model.
     * @return Gradient of the linear regression model.
     */
    protected static Matrix getGrad(Matrix X, Matrix y, Matrix w) {
        return w.T().mult(X.T()).mult(X).sub(y.T().mult(X)).T();
    }
}
