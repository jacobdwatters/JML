package com.jml.neural_network.layers.initilizers;
import com.jml.neural_network.layers.Layer;
import linalg.Matrix;

/**
 * {@link Layer} parameter initializer to produce a constant value.
 */
public class Constant implements Initializer {
    private double value;

    /**
     * Creates a Constant Initializer with value 1. This is equivalent to the {@link Ones} Initializer.
     */
    public Constant() {
        value = 0;
    }


    /**
     * Creates a Constant Initializer with specified value.
     * @param value Constant to fill matrix with.
     */
    public Constant(double value) {
        this.value = value;
    }


    /**
     * Initializes values of a weight/bias matrix as specified by this initializer.
     *
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @return A {@link Matrix} initialized with values from the distribution specified by this initializer.
     */
    @Override
    public Matrix init(int m, int n) {
        return new Matrix(m, n, value);
    }
}
