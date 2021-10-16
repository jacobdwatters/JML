package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.StochasticGradientDescent;
import com.jml.util.ValueError;
import linalg.Matrix;
import linalg.Vector;


/**
 * Model for least squares linear regression of multiple variables by stochastic gradient descent.<br><br>
 *
 * MultipleLinearRegressionSGD fits a model y = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + ... + b<sub>n</sub>x<sub>n</sub>
 * to the datasets by minimizing the residuals of the sum of squares
 * (i.e. the {@link LossFunctions#sse sum of square errors}) between the values in the
 * target dataset and the values predicted by the model. This is minimized using stochastic gradient descent.
 */
public class MultipleLinearRegressionSGD extends MultipleLinearRegression {

    private double learningRate = 0.002;
    private double threshold = 0.5e-5;
    private int maxIterations = 1000;
    private Optimizer SGD;

    /**
     * Creates a {@link MultipleLinearRegressionSGD} model.  This will use a default learning rate of 0.002.
     */
    public MultipleLinearRegressionSGD() {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(double learningRate, double threshold, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        paramCheck();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(double learningRate, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        paramCheck();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(double learningRate) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        paramCheck();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(int maxIterations) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.maxIterations = maxIterations;
        paramCheck();
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public MultipleLinearRegressionSGD fit(double[][] features, double[] targets) {
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Matrix(features));
        Matrix y = new Vector(targets);

        w = SGD.optimize(LossFunctions.sse, X, y);
        super.coefficients = w.T().getValuesAsDouble()[0];

        super.isFit=true;
        super.buildDetails();

        return this;
    }

    private void paramCheck() {
        if(!ValueError.isNonNegative(maxIterations))
            throw new IllegalArgumentException("maxIterations must be non-negative but got " + maxIterations);
        if(!ValueError.isNonNegative(learningRate))
            throw new IllegalArgumentException("learningRate must be non-negative but got " + learningRate);
        if(!ValueError.isNonNegative(threshold))
            throw new IllegalArgumentException("threshold must be non-negative but got " + threshold);
    }
}