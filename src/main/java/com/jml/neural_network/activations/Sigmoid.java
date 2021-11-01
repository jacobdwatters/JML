package com.jml.neural_network.activations;

import linalg.Matrix;

public class Sigmoid extends Activation {

    private Sigmoid() {
        throw new IllegalStateException("Cannot instantiate utility class.");
    }


    /**
     * Applies the sigmoid activation function to the data.<br>
     * Specifically, the logistic function is applied. i.e. 1/(1+e<sup>-x</sup>)
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    @Override
    public double apply(double data) {
        return 1 / (1+Math.exp(data));
    }


    /**
     * Applies the sigmoid activation function, element-wise, to the data.<br>
     * Specifically, the logistic function is applied. i.e. 1/(1+e<sup>-x</sup>)
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    @Override
    public double[] apply(double[] data) {
        double[] result = new double[data.length];

        for(int i=0; i<data.length; i++) {
            result[i] = apply(data[i]);
        }

        return result;
    }


    /**
     * Applies the sigmoid activation function, element-wise, to the data.<br>
     * Specifically, the logistic function is applied. i.e. 1/(1+e<sup>-x</sup>)
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    @Override
    public double[][] apply(double[][] data) {
        double[][] result = new double[data.length][data[0].length];

        for(int i=0; i<data.length; i++) {
            result[i] = apply(data[i]);
        }

        return result;
    }


    /**
     * Applies the sigmoid activation function, element-wise, to the data.<br>
     * Specifically, the logistic function is applied. i.e. 1/(1+e<sup>-x</sup>)
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    @Override
    public Matrix apply(Matrix data) {
        Matrix result = new Matrix(data.shape());

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
//                result.set();
            }
        }

        return result;
    }
}
