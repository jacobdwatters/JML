package com.jml.losses;

import linalg.Matrix;

/**
 * This class contains lambda functions for various loss functions including:
 * <pre>
 *     - {@link #sse}: sum of squared-errors loss.
 *     - {@link #binCrossEntropy}: binary cross-entropy loss (Cross-entropy for two classes).
 *     - {@link #crossEntropy}: cross-entropy loss (For multiple classes).
 * </pre>
 */
public class LossFunctions {

    /* TODO: All loss functions should become an object rather than a lambda so that we can define, compute() and slope() function
        for each loss. This will allow different loss functions to be specified for a NeuralNetwork since function and its
        derivative needs to be computed.
     */

    // A private constructor to hide the implicit constructor.
    private LossFunctions() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * The sum of mean squared-errors loss function.<br>
     * That is <code>mse = (1/n)*sum(x<sub>i</sub> - y<sub>i</sub>)<sup>2</sup></code>
     * where <code>x</code> and <code>y</code> are datasets of length <code>n</code>,
     * and <code>x</code> is the actual data and <code>y</code> is the predicted data.
     */
    public static final Function mse = (Matrix y, Matrix yPred) -> yPred.sub(y).T().mult(yPred.sub(y)).scalDiv(y.numRows());


    /**
     * The sum of squared-errors loss function.<br>
     * That is <code>sse = sum(x<sub>i</sub> - y<sub>i</sub>)<sup>2</sup></code>
     * where <code>x</code> and <code>y<code/> are datasets of length <code>n</code>,
     * and <code>x</code> is the actual data and <code>y</code> is the predicted data.
     */
    public static final Function sse = (Matrix y, Matrix yPred) -> yPred.sub(y).T().mult(yPred.sub(y));


    /**
     * The binary cross-entropy loss function.
     * Note: cross-entropy is undefined for p=0 or p=1, so probabilities adjusted to be "very close" to 0 or 1 if
     * appropriate.
     */
    public static final Function binCrossEntropy = (Matrix y, Matrix yPred) -> {
        double eps = 1e-15;
        double loss = 0;
        Matrix result = new Matrix(1);

        if(yPred.numCols() > 2) {
            throw new IllegalArgumentException("Predictions seem to have more than two classes. Consider " +
                    "crossEntropy() for multiple classes.");
        }

        for(int i=0; i<yPred.numRows(); i++) {
            // cross-entropy is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
            if(yPred.getAsDouble(i, 0)==0) {
                yPred.set(eps, i,  0);
            } else if(yPred.getAsDouble(i, 0)==1) {
                yPred.set(1-eps, i,  0);
            }

            loss += y.getAsDouble(i, 0)*Math.log(yPred.getAsDouble(i, 0))
                    + (1-y.getAsDouble(i, 0))*Math.log(1-yPred.getAsDouble(i, 0));
        }

        result.set(-loss/yPred.numRows(), 0,0);

        return result;
    };


    /**
     * the cross-entropy loss function.
     * Note: cross-entropy is undefined for p=0 or p=1, so probabilities adjusted to be "very close" to 0 or 1 if
     * appropriate.
     */
    public static final Function crossEntropy = (Matrix y, Matrix yPred) -> {
        double eps = 1e-15;
        double loss = 0;
        Matrix result = new Matrix(1);

        // y contains the actual labels as a one-hot vector.
        for(int i=0; i<yPred.numRows(); i++) {
            for(int j=0; j<yPred.numCols(); j++) {
                // cross-entropy is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p)).
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

        result.set(-loss/yPred.numRows(), 0,0);

        return result;
    };
}
