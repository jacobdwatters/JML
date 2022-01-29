package com.jml.neural_network.layers.initilizers;

import com.jml.neural_network.layers.Layer;
import linalg.Matrix;


/**
 * {@link Layer} parameter initializer to produce zeros.
 */
public class Zeros implements Initializer {

    /**
     * Creates a Zeros initializer.
     */
    public Zeros() {/*Does nothing*/}

    /**
     * Initialized values of a weight/bias matrix.
     *
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @return A {@link Matrix} initialized with values from the distribution specified by this initializer.
     */
    @Override
    public Matrix init(int m, int n) {
        return new Matrix(m, n);
    }
}
