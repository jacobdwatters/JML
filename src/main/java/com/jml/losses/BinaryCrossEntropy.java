package com.jml.losses;

import linalg.Matrix;


/**
 * Class for computing binary cross-entropy loss and its derivative.
 */
public class BinaryCrossEntropy implements LossFunction {
    // Small value to ensure no division by zero occurs as a result of floating point precision.
    private static final double EPS = 1e-15;
    public static final String NAME = "Binary Cross-entropy";


    @Override
    public Matrix forward(Matrix y, Matrix yPred) {
        double loss = 0;

        if(yPred.numCols() > 2) {
            throw new IllegalArgumentException("Predictions seem to have more than two classes. Consider " +
                    "CrossEntropy class for multiple classes.");
        }

        for(int i=0; i<yPred.numRows(); i++) {
            // cross-entropy is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
            if(yPred.getAsDouble(i, 0)==0) {
                yPred.set(EPS, i,  0);
            } else if(yPred.getAsDouble(i, 0)==1) {
                yPred.set(1-EPS, i,  0);
            }

            loss += y.getAsDouble(i, 0)*Math.log(yPred.getAsDouble(i, 0))
                    + (1-y.getAsDouble(i, 0))*Math.log(1-yPred.getAsDouble(i, 0));
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
