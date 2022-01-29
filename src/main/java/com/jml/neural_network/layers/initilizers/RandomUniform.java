package com.jml.neural_network.layers.initilizers;

import com.jml.neural_network.layers.Layer;
import linalg.Matrix;

import java.util.Random;


/**
 * {@link Layer} parameter initializer to produce random values from a uniform distribution.
 */
public class RandomUniform implements Initializer {
    private Random r;
    private double min = 0;
    private double max = 1;

    /**
     * Creates a RandomUniform Initializer within [0, 1].
     */
    public RandomUniform() {
        this.min = 0;
        this.max = 1;
        this.r = new Random();
    }


    /**
     * Creates a RandomUniform Initializer with specified range.
     * @param min Lower bound for random uniform distribution.
     * @param max Upper bound for random uniform distribution.
     */
    public RandomUniform(double min, double max) {
        this.min = min;
        this.max = max;
        r = new Random();
    }


    /**
     * Creates a RandomUniform Initializer with specified range and seed.
     * @param min Lower bound for random uniform distribution.
     * @param max Upper bound for random uniform distribution.
     * @param seed Seed for random number generator.
     */
    public RandomUniform(double min, double max, long seed) {
        this.min = min;
        this.max = max;
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

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                random[i][j] = min + r.nextDouble() * (max - min);;
            }
        }

        return new Matrix(random);
    }
}
