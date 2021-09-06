package com.jml.neural_network.optimizers;

abstract class Optimizer {


    /**
     * Applies optimizers update rule to x.
     *
     * @param x Value to update
     * @return
     */
    public abstract double update(int x);
}
