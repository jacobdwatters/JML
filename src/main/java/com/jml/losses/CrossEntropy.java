package com.jml.losses;

import linalg.Matrix;


/**
 * Class for computing cross-entropy loss and its derivative.
 */
public class CrossEntropy implements LossFunction {
    // Small value to ensure no division by zero occurs as a result of floating point precision.
    private static final double EPS = 1e-15;
    public static final String NAME = "Cross-entropy";


    @Override
    public Matrix forward(Matrix y, Matrix yPred) {
        double eps = 1e-15;
        double loss = 0;

        // y contains the actual labels as a one-hot vector.
        for(int i=0; i<yPred.numRows(); i++) {
            for(int j=0; j<yPred.numCols(); j++) {
                // cross-entropy is undefined for p=0 and p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
                if(yPred.getAsDouble(i, j)==0) {
                    yPred.set(eps, i,  j);
                } else if(yPred.getAsDouble(i, j)==1) {
                    yPred.set(1-eps, i,  j);
                }

                if(y.getAsDouble(i, j) == 1) {
                    loss += Math.log(yPred.getAsDouble(i, j));
                } else if(y.getAsDouble(i, j) != 0) {
                    throw new IllegalArgumentException("y does not seem to be a one-hot vector. Must contain binary entries only.");
                }
            }
        }

        return new Matrix(1, 1, loss/y.numRows());
    }


    @Override
    public Matrix back(Matrix y, Matrix yPred) {
        return yPred.sub(y).elemDiv(yPred.sub(yPred.elemMult(yPred))).scalDiv(y.numRows());
    }


    @Override
    public String getName() {
        return this.NAME;
    }
}
