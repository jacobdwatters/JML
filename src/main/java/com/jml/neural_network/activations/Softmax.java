package com.jml.neural_network.activations;

import linalg.Matrix;
import linalg.Vector;


/**
 * The softmax activation function. f(<b>x</b>)<sub>i</sub> = exp(<b>x</b>_i) / sum<sub>j=1</sub><sup>m</sup>( exp(<b>x</b><sub>j</sub>) ) where
 * <b>x</b> is a vector of length m and sum<sub>j=1</sub><sup>m</sup> ( exp(<b>x</b><sub>j</sub>) ) = exp(<b>x</b><sub>1</sub>) + exp(<b>x</b><sub>2</sub>) + ... + exp(<b>x</b><sub>m</sub>).
 */
public class Softmax implements ActivationFunction {

    public final String NAME = "Softmax";

    /**
     * Applies the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply activation function to.
     * @return The result of the element-wise activation function applied to the matrix.
     */
    @Override
    public Matrix apply(Matrix data) {
        if(data.numCols()>1) {
            throw new IllegalArgumentException("Expecting column matrix for " + NAME + " activation but got shape " + data.shape());
        }

        double[] result = new double[data.numRows()];
        double sum = 0;

        for(int i=0; i<data.numRows(); i++) { // Compute denominator.
            sum += Math.exp(data.getAsDouble(i, 0));
        }

        for(int i=0; i<data.numRows(); i++) { // Compute each entry for the resulting vector.
            result[i] = Math.exp(data.getAsDouble(i, 0)) / sum;
        }

        return new Vector(result); // Return Softmax activation result as a column vector.
    }

    /**
     * Applies the derivative of the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply the derivative of the activation function to.
     * @return The slope of the activation function, evaluated element-wise, of the data matrix.
     */
    @Override
    public Matrix slope(Matrix data) {
        return null;
    }

    /**
     * Gets the name of the activation function.
     *
     * @return The name of the activation function as a String.
     */
    @Override
    public String getName() {
        return this.NAME;
    }
}
