package com.jml.losses;


import linalg.Matrix;

/**
 * This class contains methods for computing the loss between two datasets stored in double[][] arrays.
 */
public abstract class Loss {


    /**
     * Computes the of mean sum of squared-errors loss function.<br>
     * That is <code>mse = (1/n)*sum(yPred<sub>i</sub> - y<sub>i</sub>)<sup>2</sup></code>.
     *
     * @param y Expected value.
     * @param yPred Predicted value.
     * @return Mean squared-error loss of y and yPred.
     */
    public static double mse(double[][] y, double[][] yPred) {
        return LossFunctions.mse.compute(new Matrix(y), new Matrix(yPred)).getAsDouble(0, 0);
    }


    /**
     * Computes the of sum of squared-errors loss function.<br>
     * That is <code>sse = sum(x<sub>i</sub> - y<sub>i</sub>)<sup>2</sup></code>.
     *
     * @param y Expected value.
     * @param yPred Predicted value.
     * @return Squared-error loss of y and yPred.
     */
    public static double sse(double[][] y, double[][] yPred) {
        return LossFunctions.sse.compute(new Matrix(y), new Matrix(yPred)).getAsDouble(0, 0);
    }


    /**
     * The binary cross-entropy loss function. bce = -(1/n) * sum( y*<sub>i</sub>ln(yPred<sub>i</sub>) + (1-y<sub>i</sub>)*ln(yPred<sub>i</sub>) )<br><br>
     * Note: binary cross-entropy is undefined for p=0 or p=1, so probabilities adjusted to be "very close" to 0 or 1 if
     * appropriate.
     *
     * @param y Expected value.
     * @param yPred Predicted value.
     * @return Squared-error loss of y and yPred.
     */
    public static double binCrossEntropy(double[][] y, double[][] yPred) {
        return LossFunctions.binCrossEntropy.compute(new Matrix(y), new Matrix(yPred)).getAsDouble(0, 0);
    }


    /**
     * The binary cross-entropy loss function. ce = -(1/n) * sum( y*<sub>i</sub>ln(yPred<sub>i</sub>) )<br><br>
     * Note: cross-entropy is undefined for p=0 or p=1, so probabilities adjusted to be "very close" to 0 or 1 if
     * appropriate.
     *
     * @param y Expected value.
     * @param yPred Predicted value.
     * @return Squared-error loss of y and yPred.
     */
    public static double crossEntropy(double[][] y, double[][] yPred) {
        return LossFunctions.crossEntropy.compute(new Matrix(y), new Matrix(yPred)).getAsDouble(0, 0);
    }
}
