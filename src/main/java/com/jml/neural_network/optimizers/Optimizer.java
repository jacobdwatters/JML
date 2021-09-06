package com.jml.neural_network.optimizers;

abstract class Optimizer {


    /**
     * Applies specified optimizer rule to x.
     *
     * @param weights Weights of layer
     * @param outputs output of layer
     * @return
     */
    public abstract double optimize(double[][] weights, double[] outputs);
}
