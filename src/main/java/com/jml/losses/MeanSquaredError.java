package com.jml.losses;

import linalg.Matrix;


/**
 * Class for computing the mean squared error loss and its derivative.<br><br>
 *
 * <code>mse = (1/n)*sum(x<sub>i</sub> - y<sub>i</sub>)<sup>2</sup></code>
 * where <code>x</code> and <code>y</code> are datasets of length <code>n</code>,
 * and <code>x</code> is the actual data and <code>y</code> is the predicted data.
 */
public class MeanSquaredError implements LossFunction {
    public static final String NAME = "Mean Squared Error";


    @Override
    public Matrix forward(Matrix y, Matrix yPred) {
        return yPred.sub(y).T().mult(yPred.sub(y)).scalDiv(y.numRows());
    }


    @Override
    public Matrix back(Matrix y, Matrix yPred) {
        return yPred.sub(y);
    }


    @Override
    public String getName() {
        return this.NAME;
    }
}
