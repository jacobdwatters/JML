package com.jml.neural_network.layers.initilizers;

import com.jml.neural_network.layers.Layer;
import linalg.Matrix;


/**
 * {@link Layer} parameter initializer to produce ones.
 */
public class Ones implements Initializer {

    /**
     * Creates a Ones initializer.
     */
    public Ones() {/*Does nothing*/}

    /**
     * Initializes values of a weight/bias matrix as specified by this initializer.
     *
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @return A {@link Matrix} initialized with values from the distribution specified by this initializer.
     */
    @Override
    public Matrix init(int m, int n) {
        return Matrix.ones(m, n);
    }
}
