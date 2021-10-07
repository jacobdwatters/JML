package com.jml.optimizers;


import com.jml.losses.Function;
import linalg.Matrix;

abstract class Optimizer {

    /**
     * Applies specified optimizer rule to x.
     *
     * @return The weights which optimize
     */
    public abstract Matrix optimize(Matrix w, Matrix X, Matrix y, Function lossGrad, int alpha);


    /**
     * Applies specified optimizer rule to x.
     *
     * @return The weights which optimize
     */
    public abstract Matrix optimize(Matrix w, Matrix X, Matrix y, Function lossGrad, int alpha, int maxIterations);
}
