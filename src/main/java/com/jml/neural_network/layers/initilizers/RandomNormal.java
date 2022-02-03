package com.jml.neural_network.layers.initilizers;

import linalg.Matrix;
import java.util.Random;


/**
 * {@link com.jml.neural_network.layers.TrainableLayer layer} parameter initializer to produce random values from a normal distribution.
 */
public class RandomNormal implements Initializer {

    private Random r;
    private double mean;
    private double std;


    /**
     * Creates a RandomNormal initializer with mean 0 and standard deviation 1.
     */
    public RandomNormal() {
        this.mean = 0;
        this.std = 1;
        this.r = new Random();
    }


    /**
     * Creates a RandomNormal initializer with mean 0 and specified standard deviation.
     * @param std Standard deviation for random normal distribution.
     */
    public RandomNormal(double std) {
        this.mean = 0;
        this.std = std;
        this.r = new Random();
    }


    /**
     * Creates a RandomNormal initializer with specified mean and standard deviation.
     * @param std Standard deviation for random normal distribution.
     * @param mean Mean for random normal distribution.
     */
    public RandomNormal(double std, double mean) {
        this.mean = mean;
        this.std = std;
        this.r = new Random();
    }


    /**
     * Creates a RandomNormal initializer with specified mean and standard deviation.
     * @param std Standard deviation for random normal distribution.
     * @param mean Mean for random normal distribution.
     * @param seed Seed for the random number generator.
     */
    public RandomNormal(double std, double mean, long seed) {
        this.mean = 0;
        this.std = std;
        this.r = new Random(seed);
    }


    /**
     * Creates and initializes a matrix with entries from a Random Normal Distribution.
     * @param m Number of rows in the matrix to initialize.
     * @param n Number of columns in the matrix.
     * @return
     */
    @Override
    public Matrix init(int m, int n) {
        double[][] random = new double[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                random[i][j] = r.nextGaussian()*std + mean;
            }
        }

        return new Matrix(random);
    }
}
