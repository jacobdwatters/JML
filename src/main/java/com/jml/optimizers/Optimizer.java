package com.jml.optimizers;

import com.jml.losses.Function;
import linalg.Matrix;

public abstract class Optimizer {

    /**
     * Applies specified optimizer rule to loss function.
     *
     * @param loss The loss function to optimize.
     * @param X Features of model.
     * @param y Targets of model.
     * @return The computed minimum of the loss function.
     */
    public abstract Matrix optimize(Function loss, Matrix X, Matrix y);
}
