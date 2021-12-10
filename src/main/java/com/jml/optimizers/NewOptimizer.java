package com.jml.optimizers;

import com.jml.core.Model;
import linalg.Matrix;

// TODO: This should eventually replace the Optimizer class.
public abstract class NewOptimizer {
    double learningRate = 0.02; // Learning rate of the optimizer.
    int iterations = 0; // The number of times step() has been applied for this optimizer.
    Model model; // Model which the optimizer is working on. This is needed since loss functions depends on a model.
    Scheduler schedule; // Learning rate scheduler rule to apply during optimization. If this is left as null, then no rule will be applied.

    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.
     *
     * @param w A matrix contain the
     * @param wGrad The gradient of w.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    public abstract Matrix step(Matrix w, Matrix wGrad);


    /**
     * Sets the learning rate scheduler for this optimizer.
     * @param schedule Learning rate scheduler.
     */
    public void setScheduler(Scheduler schedule) {
        this.schedule = schedule;
    }
}
