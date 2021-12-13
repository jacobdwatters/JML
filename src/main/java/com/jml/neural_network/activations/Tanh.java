package com.jml.neural_network.activations;


import linalg.Matrix;

/**
 * The hyperbolic tangent activation function.
 * <code>f(x) = tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</code>
 */
public class Tanh implements ActivationFunction{
    public static final String NAME = "tanh";


    /**
     * Applies the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply activation function to.
     * @return The result of the element-wise activation function applied to the matrix.
     */
    @Override
    public Matrix apply(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];
        double exp, negExp;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                exp = Math.exp(data.getAsDouble(i, j));
                negExp = Math.exp(data.getAsDouble(i, j));

                result[i][j] =  (exp - negExp) / (exp + negExp);
            }
        }

        return new Matrix(result);
    }


    /**
     * Applies the derivative of the activation function, element-wise, to a matrix.
     *
     * @param data The matrix to apply the derivative of the activation function to.
     * @return
     */
    @Override
    public Matrix slope(Matrix data) {
        double[][] result = new double[data.numRows()][data.numCols()];
        double exp, negExp;

        for(int i=0; i<data.numRows(); i++) {
            for(int j=0; j<data.numCols(); j++) {
                exp = Math.exp(data.getAsDouble(i, j));
                negExp = Math.exp(data.getAsDouble(i, j));

                result[i][j] =  4 / Math.pow(exp + negExp, 2);
            }
        }

        return new Matrix(result);
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
