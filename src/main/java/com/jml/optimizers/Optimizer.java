package com.jml.optimizers;

import linalg.Matrix;

/**
 * Interface for an optimizer which can be used to minimize a function by utilizing the gradients of that function.
 */
public abstract class Optimizer {
    double learningRate; // Learning rate of the optimizer.
    public String name;

    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.<br><br>
     *
     * WARNING: If this step method is called for the {@link Momentum} optimizer an exception will be thrown.
     * Use {@link #step(Matrix, Matrix, Matrix)} instead.
     *
     * @param w A matrix containing the weights to apply the update to.
     * @param wGrad The gradient of w with respect to some function (Most likely a model).
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    public abstract Matrix step(Matrix w, Matrix wGrad);


    /**
     * Steps the optimizer a single iteration by applying the update rule of the optimizer to the matrix w. This
     * step method should be used for momentum.
     *
     * @param w A matrix containing the weights to apply the update to.
     * @param wGrad The gradient of w with respect to some function (Most likely a model).
     * @param v The update vector for the momentum optimizer. If the optimizer is {@link GradientDescent} This will have
     *          no effect.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    public abstract Matrix[] step(Matrix w, Matrix wGrad, Matrix v);


    /**
     * Gets the details of this optimizer.
     * @return Important details of this optimizer as a string.
     */
    public abstract String getDetails();


    /**
     * Gets the learning rate for this optimizer.
     * @return The learning rate for this optimizer.
     */
    public double getLearningRate() {return learningRate;}
}
