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


    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.<br><br>
     * <p>
     * WARNING: If this step method is called for the {@link Momentum} optimizer an exception will be thrown.
     * Use {@link #step(Matrix, Matrix, Matrix)} instead.
     *
     * @param w     A matrix containing the weights to apply the update to.
     * @param wGrad The gradient of w with respect to some function (Most likely a model).
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    @Override
    public Matrix step(Matrix w, Matrix wGrad) {
        // TODO: Auto-generated method stub
        return null;
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
        // TODO: Auto-generated method stub
        return new Matrix[0];
    }
}
