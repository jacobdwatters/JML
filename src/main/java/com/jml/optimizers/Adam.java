package com.jml.optimizers;

import linalg.Matrix;


/**
 * The Adam (Adaptive Moment Estimation) optimizer. Adam is a first order gradient method for efficient
 * stochastic optimization.<br><br>
 *
 * This is recommended as the default optimizer on most problems.
 *
 * Applies the following update rule:
 * <pre>
 *      g<sub>t</sub> = grad(w<sub>t-1</sub>)
 *      m<sub>t</sub> = b<sub>1</sub>*m<sub>t-1</sub> + (1-b<sub>1</sub>)*g<sub>t</sub>
 *      v<sub>t</sub> = b<sub>2</sub>*v<sub>t-1</sub> + (1-b<sub>2</sub>)*g<sub>t</sub><sup>2</sup>
 *
 *      m&#770;<sub>t</sub> = m<sub>t</sub>/(1-b<sub>1</sub><sup>t</sup>)
 *      v&#770;<sub>t</sub> = v<sub>t</sub>/(1-b<sub>2</sub><sup>t</sup>)
 *
 *      w<sub>t</sub> = w<sub>t-1</sub> - a*m&#770;<sub>t</sub>/(&#8730;(v&#770;<sub>t</sub>) + &#1013;)
 *
 *      Where a is the learning rate, b<sub>1</sub> and b<sub>2</sub> are the moment parameters,
 *      and &#1013; is a small value to avoid division bny zero.
 * </pre>
 *
 *
 */
public class Adam extends Optimizer {

    final static double eps = 1E-7;
    double beta1, beta2;
    double t; // Time step
    double alpha;
    public static final String OPTIM_NAME = "Adam";

    /**
     * Creates a Adam optimizer with specified learning rate and parameters.
     *
     * @param learningRate Learning rate for the Adam optimizer.
     * @param beta1 Exponential decay rate for first moment estimate. Must be in [0, 1)
     * @param beta2 Exponential decay rate for second moment estimate. Must be in [0, 1)
     */
    public Adam(double learningRate, double beta1, double beta2) {
        super.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        t=0;
        super.name = OPTIM_NAME;
    }


    /**
     * Steps the optimizer a single iteration by applying the update rule of the optimizer to the matrix w.
     * Note, this will increase the time step of the Adam optimizer. To not increase the time step see
     * {@link #step(boolean, Matrix[])}.
     * @param params An array of matrices strictly containing {w, wGrad, v, m}
     *               where w is the matrix containing the weights to apply the update to,
     *               wGrad is the gradient of the objective function with respect to w,
     *               v is the second moment estimate, and v is the first moment estimate.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    public Matrix[] step(Matrix... params) {
        return step(true, params);
    }


    /**
     * Steps the optimizer a single iteration by applying the update rule of the optimizer to the matrix w.
     *
     * @param increaseTime Flag for increasing the timestamp of the Adam optimizer.
     * @param params An array of matrices strictly containing {w, wGrad, v, m}
     *               where w is the matrix containing the weights to apply the update to,
     *               wGrad is the gradient of the objective function with respect to w,
     *               v is the second moment estimate, and v is the first moment estimate.
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    public Matrix[] step(boolean increaseTime, Matrix... params) {
        if(params.length != 4) {
            throw new IllegalArgumentException("Step method for " + OPTIM_NAME +
                    " expecting 4 matrices but got " + params.length);
        }

        Matrix w = params[0];
        Matrix wGrad = params[1];
        Matrix v = params[2];
        Matrix m = params[3];

        if(increaseTime) {
            t++;
        }

        m = m.scalMult(beta1).add(wGrad.scalMult(1-beta1));
        v = v.scalMult(beta2).add(wGrad.elemMult(wGrad).scalMult(1-beta2));

        alpha = learningRate*Math.sqrt(1-Math.pow(beta2, t)) / (1-Math.pow(beta1, t));
        w = w.sub(m.scalMult(learningRate).elemDiv(v.sqrt().add(eps)));

        return new Matrix[]{w, v, m};
    }


    /**
     * Gets the details of this optimizer.
     *
     * @return Important details of this optimizer as a string.
     */
    @Override
    public String getDetails() {
        // TODO:
        return null;
    }
}
