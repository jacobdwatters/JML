package com.jml.neural_network.layers.initilizers;

import linalg.Matrix;


/**
 * Interface for bias/weight initialization.
 * Available Initializer:
 * <pre>
 *     - {@link RandomNormal}
 *     - {@link RandomUniform}
 *     - {@link Zeros}
 *     - {@link Ones}
 *     - {@link Constant}
 *     - {@link Orthogonal}</pre>
 */
public interface Initializer {

    /**
     * Initializes values of a weight/bias matrix as specified by this initializer.
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @return A {@link linalg.Matrix} initialized with values from the distribution specified by this initializer.
     */
    Matrix init(int m, int n);
}
