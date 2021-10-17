package com.jml.linear_models;


import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.Scheduler;
import com.jml.optimizers.StochasticGradientDescent;
import linalg.Matrix;
import linalg.Vector;
import com.jml.util.ValueError;


/**
 * Model for least squares linear regression of one variable by stochastic gradient descent.<br><br>
 *
 * LinearRegressionSGD fits a model y = b<sub>0</sub> + b<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is using stochastic gradient descent.
 */
public class LinearRegressionSGD extends LinearRegression {

    protected double learningRate = 0.002;
    protected double threshold = 0.5e-5;
    protected int maxIterations = 1000;
    protected Optimizer SGD;
    protected Scheduler schedule;


    /**
     * Creates a {@link LinearRegressionSGD} model.<br>
     * This will use default settings for gradient descent:
     * <pre>
     *    Learning Rate: 0.002
     *    Threshold: 0.5e-5
     *    Maximum Iterations: 1000
     *    Scheduler: None
     * <pre/>
     */
    public LinearRegressionSGD() {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * @param schedule Learning rate scheduler.
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate, int maxIterations, double threshold, Scheduler schedule) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.schedule = schedule;
        paramCheck();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate, int maxIterations, double threshold) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        paramCheck();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        paramCheck();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        paramCheck();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(int maxIterations) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.maxIterations = maxIterations;
        paramCheck();
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
    public LinearRegressionSGD fit(double[] features, double[] targets) {
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);
        SGD.setScheduler(this.schedule);

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Vector(features));
        Matrix y = new Vector(targets);

        w = SGD.optimize(LossFunctions.sse, X, y);

        // Update the model details
        super.isFit=true;
        buildDetails();

        return this;
    }


    /**
     * Gets the loss history from training.
     *
     * @return The loss of every iteration stored in a List.
     */
    public double[] getLossHist() {
        if(!isFit) {
            throw new IllegalStateException("Model must be trained before the loss history can be computed.");
        }

        return SGD.getLossHist().stream().mapToDouble(Double::doubleValue).toArray();
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
