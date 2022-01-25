package com.jml.optimizers;

import linalg.Matrix;

/**
 * Interface for an optimizer which can be used to minimize a function by utilizing the gradients of that function.
 */
public abstract class Optimizer {
    double learningRate; // Learning rate of the optimizer.
    public Scheduler schedule; // Learning rate scheduler rule to apply during optimization. If this is left as null, then no rule will be applied.
    public String name; // Name of the optimizer.

    // TODO: Replace all step(...) methods with the following two.
//    public abstract Matrix[] step(Matrix... args);
//    public abstract Matrix[] step(boolean flag, Matrix... args);


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
     * Steps the optimizer a single iteration by applying the update rule of the optimizer to the matrix w. This step
     * method should only be used for Adam.
     *
     * @param w A matrix containing the weights to apply the update rule to.
     * @param wGrad The gradient of the function with respect to w.
     * @param v Vector to hold first moment estimations.
     * @param m Vector to hold second moment estimations.
     * @param increaseTime If true, increase time step for optimizer. If false, do nothing.
     * @return The result of applying the update rule of the optimizer to w.
     */
    public abstract Matrix[] step(Matrix w, Matrix wGrad, Matrix v, Matrix m, boolean increaseTime);


    /**
     * Gets the details of this optimizer.
     * @return Important details of this optimizer as a string.
     */
    public abstract String getDetails();


    /**
     * Sets the learning rate scheduler for this optimizer.
     * @param schedule Learning rate scheduler.
     */
    public void setScheduler(Scheduler schedule) {
        this.schedule = schedule;
    }


    /**
     * Gets the learning rate for this optimizer.
     * @return The learning rate for this optimizer.
     */
    public double getLearningRate() {return learningRate;}
}
