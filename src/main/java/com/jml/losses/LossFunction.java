package com.jml.losses;

import linalg.Matrix;


/**
 * Interface for loss functions.
 */
public interface LossFunction {


    /**
     * Computes the forward pass of the loss function. That is, evaluates the loss function at specified values.
     *
     * @param y Expected values.
     * @param yPred Predicted values to use in loss computation.
     * @return Result of loss function evaluated at specified positions.
     */
    Matrix forward(Matrix y, Matrix yPred);


    /**
     * Computes the backward pass of the loss function. That is, computes the derivative of the loss function
     * evaluated at a certain point.
     *
     * @param y Expected values.
     * @param yPred Predicted values to use in loss computation.
     * @return Loss derivative evaluated at specified points.
     */
    Matrix back(Matrix y, Matrix yPred);


    /**
     * Gets the name of the loss function.
     * @return Name of the loss function as a String.
     */
    String getName();
}
