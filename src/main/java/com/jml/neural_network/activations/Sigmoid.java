package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * The sigmoid activation function. 1/(1+exp(-x))
 */
class Sigmoid implements ActivationFunction {

    public final String NAME = "Sigmoid";

    /**
     * Applies the sigmoid activation element-wise to a matrix.
     *
     * @param data Matrix to apply sigmoid activation to.
     * @return The result of the sigmoid activation applied element-wise to the data matrix.
     */
    @Override
    public Matrix apply(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                result[i][j] = 1/(1+Math.exp(-data.getAsDouble(i, j)));
            }
        }

        return new Matrix(result);
    }


    /**
     * Applies the derivative of the sigmoid activation function element-wise to a matrix.
     *
     * @param data Matrix to apply the derivative of the sigmoid activation function to.
     * @return The result of the derivative of the sigmoid activation applied element-wise to the data.
     */
    @Override
    public Matrix slope(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];
        double exp;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                exp = Math.exp(-data.getAsDouble(i, j));
                result[i][j] = exp/Math.pow(1+exp, 2);
            }
        }

        return new Matrix(result);
    }


    /**
     * Gets the name of the activation function.
     * @return The name of the activation function as a String.
     */
    @Override
    public String getName(){return NAME;}
}
