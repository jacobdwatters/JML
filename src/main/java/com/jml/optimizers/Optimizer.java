package com.jml.optimizers;

import linalg.Matrix;

/**
 * Interface for an optimizer which can be used to minimize a function by utilizing the gradients of that function.
 */
public abstract class Optimizer {
    double learningRate; // Learning rate of the optimizer.
    public Scheduler schedule; // Learning rate scheduler rule to apply during optimization. If this is left as null, then no rule will be applied.
    public String name; // Name of the optimizer.

    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.
     *
     * @param params An array of Matrices strictly containing the relevant parameters for that optimizer.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    public abstract Matrix[] step(Matrix... params);
    public abstract Matrix[] step(boolean flag, Matrix... params);


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
