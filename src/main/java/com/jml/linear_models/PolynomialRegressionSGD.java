package com.jml.linear_models;

import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.StochasticGradientDescent;
import linalg.Matrix;
import linalg.Vector;


public class PolynomialRegressionSGD extends PolynomialRegression {
    private double learningRate = 0.002;
    private double threshold = 0.5e-5;
    private int maxIterations = 1000;
    private Optimizer SGD;


    /**
     * Creates a {@link PolynomialRegressionSGD} model. This will use a default learning rate of 0.002.
     */
    public PolynomialRegressionSGD() {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(int degree, double learningRate, double threshold, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(double learningRate, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(double learningRate) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(int maxIterations) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.maxIterations = maxIterations;
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return Returns details of the fitting / training process.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    @Override
    public PolynomialRegressionSGD fit(double[] features, double[] targets) {
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.van( new Vector(features), degree+1); // TODO;
        Matrix y = new Vector(targets);

        w = SGD.optimize(LossFunctions.sse, X, y);

        // Update the model details
        super.isFit=true;
        buildDetails();

        return this;
    }
}
