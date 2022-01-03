package com.jml.optimizers;


/**
 * Learning rate scheduler for optimizers. For many models which are fit using an {@link Optimizer}, it may be beneficial
 * to decrease the learning rate according to some rule as the optimizer iterates. This interface provides the ability to
 * define a rule to accomplish this.
 */
public abstract class Scheduler {

    // TODO: Add minLearningRate which will specify a lower bound for the learning rate.

    /**
     * Optimizer to apply scheduler to.
     */
    Optimizer optim;

    /**
     * Applies the specified scheduler rule to the learning rate of the optimizer.
     */
    public abstract void step();


    /**
     * Sets the {@link Optimizer} for this Scheduler. This will replace the current optimizer of this
     * Scheduler.
     *
     * @param optim {@link Optimizer} for this Scheduler.
     */
    public void setOptim(Optimizer optim) {
        this.optim = optim;
    }


    /**
     * Gets the {@link Optimizer} for this Scheduler.
     * @return The optimizer of this Scheduler.
     */
    public Optimizer getOptim() {
        return this.optim;
    }
}
