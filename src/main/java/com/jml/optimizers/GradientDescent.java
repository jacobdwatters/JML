package com.jml.optimizers;

import linalg.Matrix;


/**
 * Vanilla gradient descent optimizer. Applies the update rule,
 * <code>w<sub>i+1</sub> = w<sub>i</sub> - a*grad( w<sub>i</sub> )</code> where <code>a</code> is the learning rate.
 */
public class GradientDescent  extends NewOptimizer {

    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.
     *
     * @param w A matrix contain the
     * @param wGrad The gradient of w.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    @Override
    public Matrix step(Matrix w, Matrix wGrad) {
        return w.sub(wGrad.scalMult(learningRate));
    }
}
