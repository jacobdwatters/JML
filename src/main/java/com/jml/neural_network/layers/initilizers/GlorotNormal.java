package com.jml.neural_network.layers.initilizers;

import linalg.Matrix;

import java.util.Random;


/**
 * {{@link com.jml.neural_network.layers.TrainableLayer layer} parameter initializer to produce random values from a truncated normal distribution with standard deviation
 * std = sqrt(2 / (fanIn + fanOut)) and mean of 0 where fanIn is the input dimension for the layer and fanOut is the output dimension for
 * the layer.
 */
public class GlorotNormal implements Initializer {
    private Random r;


    /**
     * Creates a GlorotNormal initializer with specified seed.
     */
    public GlorotNormal() {
        r = new Random();
    }


    /**
     * Creates a GlorotNormal initializer with specified seed.
     * @param seed Seed for the random number generator.
     */
    public GlorotNormal(long seed) {
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
        double std = Math.sqrt(2.0/(m+n));

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                random[i][j] = r.nextGaussian()*std;
            }
        }

        return new Matrix(random);
    }
}
