package com.jml.neural_network.activations;

import linalg.Matrix;


/**
 * The ReLU (Rectified Linear Unit) activation function. That is f(x)=max(0, x)
 */
class Relu implements ActivationFunction {

    public static final String NAME = "ReLU";

    /**
     * Applies the ReLU activation function to a matrix element-wise.
     *
     * @param data Matrix to apply ReLU activation function to.
     * @return The result of the ReLU activation function applied to the data matrix.
     */
    @Override
    public Matrix forward(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];
        double value;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                value = data.getAsDouble(i, j);
                if(value > 0) {
                    result[i][j] = value;
                } else {
                    result[i][j] = 0;
                }
            }
        }

        return new Matrix(result);
    }


    /**
     * Applies the derivative of the ReLU activation function to a matrix element-wise.
     *
     * @param data Matrix to apply the derivative of the ReLU activation function to.
     * @return The result of the derivative of the ReLU activation function applied to the data matrix.
     */
    @Override
    public Matrix back(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];
        double value;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                if(data.getAsDouble(i, j) > 0) {
                    result[i][j] = 1;
                } else {
                    result[i][j] = 0;
                }
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
