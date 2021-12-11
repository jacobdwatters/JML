package com.jml.optimizers;


/**
 * Learning rate scheduler for optimizers. For many models which are fit using an {@link Optimizer}, it may be beneficial
 * to decrease the learning rate according to some rule as the optimizer iterates. This interface provides the ability to
 * define a rule to accomplish this.
 */
public abstract class Scheduler {

    /**
     * Applies the specified scheduler rule to the learning rate of the optimizer.
     *
     * @param optm Optimizer to apply this Scheduler to.
     */
    public abstract void step(Optimizer optm);
}
