package com.jml.optimizers;

import linalg.Matrix;


/**
 * Vanilla gradient descent optimizer. Applies the update rule,
 * <code>w<sub>i+1</sub> = w<sub>i</sub> - a*grad( w<sub>i</sub> )</code> where <code>a</code> is the learning rate.
 */
public class GradientDescent  extends Optimizer {
    public static final String OPTIM_NAME = "Gradient Descent";


    /**
     * Creates a vanilla gradient descent optimizer with the specified learning rate.
     *
     * @param learningRate Learning rate to be used in the gradient descent algorithm.
     */
    public GradientDescent(double learningRate) {
        if(learningRate < 0) {
            throw new IllegalArgumentException("Learning rate must be non-negative but got " + learningRate + ".");
        }

        this.learningRate = learningRate;
        super.name = this.OPTIM_NAME;
    }

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

    /**
     * Steps the optimizer a single iteration by applying the update rule of the optimizer to the matrix w. This
     * step method should be used for momentum.
     *
     * @param w     A matrix containing the weights to apply the update to.
     * @param wGrad The gradient of w with respect to some function (Most likely a model).
     * @param v     The update vector for the momentum optimizer. If the optimizer is {@link GradientDescent} This will have
     *              no effect.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    @Override
    public Matrix[] step(Matrix w, Matrix wGrad, Matrix v) {
        throw new IllegalStateException("This step method is not defined for the " + OPTIM_NAME + " optimizer");
    }


    /**
     * Gets the details of this optimizer.
     *
     * @return Important details of this optimizer as a string.
     */
    @Override
    public String getDetails() {
        StringBuilder details = new StringBuilder();

        // TODO:

        return details.toString();
    }
}
