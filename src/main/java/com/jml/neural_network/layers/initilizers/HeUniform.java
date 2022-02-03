package com.jml.neural_network.layers.initilizers;

import linalg.Matrix;

import java.util.Random;


/**
 * {@link com.jml.neural_network.layers.TrainableLayer layer} parameter initializer to produce random values from a clipped uniform distribution in [-lim, lim] where
 * lim = sqrt(6 / fanIn), fanIn is the input dimension for the layer.
 */
public class HeUniform implements Initializer {
    private Random r;
    private double lim;


    /**
     * Creates a HeUniform initializer.
     */
    public HeUniform() {
        r = new Random();
    }


    /**
     * Creates a HeUniform initializer.
     * @param seed Seed for random number generator.
     */
    public HeUniform(long seed) {
        r = new Random(seed);
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
        double[][] random = new double[m][n];
        double lim = Math.sqrt(6.0/n);

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                random[i][j] = (-lim) + r.nextDouble() * (2*lim);;
            }
        }

        return new Matrix(random);
    }
}
