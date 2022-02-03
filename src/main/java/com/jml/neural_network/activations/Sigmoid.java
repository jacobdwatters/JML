package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * The sigmoid activation function. f(x) = 1/(1+exp(-x))
 */
public class Sigmoid implements ActivationFunction {

    public final String NAME = "Sigmoid";
    private final double LARGE_VALUE = 200;

    /**
     * Applies the sigmoid activation element-wise to a matrix.
     *
     * @param data Matrix to apply sigmoid activation to.
     * @return The result of the sigmoid activation applied element-wise to the data matrix.
     */
    @Override
    public Matrix forward(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];
        double x;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                x = data.getAsDouble(i, j);

                if(x<-LARGE_VALUE) {
                    x = -LARGE_VALUE;
                } else if(x>LARGE_VALUE) {
                    x = LARGE_VALUE;
                }

                result[i][j] = 1/(1+Math.exp(-x));
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
    public Matrix back(Matrix data) {
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
