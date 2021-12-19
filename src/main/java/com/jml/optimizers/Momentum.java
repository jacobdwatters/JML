package com.jml.optimizers;


import linalg.Matrix;

/**
 * The momentum based gradient descent optimizer. This is similar to vanilla gradient descent but will dampen oscillations
 * during optimization by helping accelerate the optimization in the relevant direction where the gradient may be steeper
 * than in other directions.<br><br>
 *
 * Applies the following update rule:
 * <pre>
 *      <code>v<sub>t</sub> = y*v<sub>t-1</sub> + a*grad( w<sub>t</sub> )</code>
 *      <code>w<sub>t+1</sub> = w<sub>t</sub> - v<sub>t</sub></code>
 *
 *      Where <code>a</code> is the learning rate and <code>y</code> is the momentum with <code>0 <= y <= 1</code>
 * </pre>
 *
 * Note: If the momentum term is zero, then this optimizer is equivalent to {@link GradientDescent Vanilla Gradient Descent}.
 */
public class Momentum extends Optimizer {
    public static final String OPTIM_NAME = "Momentum";
    double momentum = 0;


    /**
     * Creates a Momentum optimizer with specified learning rate. A default momentum of 0.9 will be used. To
     * specify a momentum, see {@link #Momentum(double, double)}.
     *
     * @param learningRate
     */
    public Momentum(double learningRate) {
        if(learningRate < 0) {
            throw new IllegalArgumentException("Learning rate must be non-negative but got " + learningRate + ".");
        }

        super.learningRate = learningRate;
        super.name = this.OPTIM_NAME;
        this.momentum=0.9;
    }


    /**
     * Creates a Momentum optimizer with specified learning rate and momentum.
     *
     * @param learningRate Learning rate to use when applying this optimizer.
     * @param momentum Momentum value to use when applying this optimizer.
     */
    public Momentum(double learningRate, double momentum) {
        if(momentum > 1 || momentum < 0) {
            throw new IllegalArgumentException("Momentum must be between 0 and 1 inclusive but got " + momentum + ".");
        }
        if(learningRate < 0) {
            throw new IllegalArgumentException("Learning rate must be non-negative but got " + learningRate + ".");
        }

        super.learningRate = learningRate;
        super.name = this.OPTIM_NAME;
        this.momentum=momentum;
    }


    /**
     * Steps the optimizer a single iteration by applying the update rule of
     * the optimizer to the matrix w.
     *
     * @param w     A matrix contain the
     * @param wGrad The gradient of w with respect to some function (Most likely a model).
     * @return The result of applying the update rule of the optimizer to the matrix w.
     */
    @Override
    public Matrix step(Matrix w, Matrix wGrad) {
        throw new IllegalArgumentException("This step method is not defined for momentum.");
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
        v = v.scalMult(momentum).add(wGrad.scalMult(learningRate));
        return new Matrix[]{w.sub(v), v};
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
