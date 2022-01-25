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
        super.name = OPTIM_NAME;
    }


    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.
     *
     * @param params An array of Matrices strictly containing {w, wGrad}
     * where w is a matrix contain the weights to apply the update to
     * and wGrad is the gradient of the objective function with respect to w.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    @Override
    public Matrix[] step(Matrix... params) {
        if(params.length != 2) {
            throw new IllegalArgumentException("Step method for " + OPTIM_NAME +
                    " expecting two matrices but got " + params.length);
        }

        Matrix w = params[0];
        Matrix wGrad = params[1];
        return new Matrix[]{w.sub(wGrad.scalMult(learningRate))};
    }


    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.
     *
     * @param flag Does nothing for gradient descent optimizer.
     * @param params An array of Matrices strictly containing {w, wGrad}
     * where w is a matrix contain the weights to apply the update to
     * and wGrad is the gradient of the objective function with respect to w.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    @Override
    public Matrix[] step(boolean flag, Matrix... params) {
        return step(params);
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
