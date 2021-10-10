package com.jml.optimizers;

/**
 * Learning rate scheduler for optimizers. For many models which are fit using an {@link Optimizer}, it may be beneficial
 * to decrease the learning rate according to some rule as the optimizer iterates. This class provides the ability to
 * define a rule to accomplish this.
 */
public interface Scheduler {
    public double apply(double learningRate);
}
