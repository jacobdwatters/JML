package com.jml.linear_models;


import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.*;
import com.jml.util.ArrayUtils;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.List;


/**
 * Model for least squares linear regression of one variable by stochastic gradient descent.<br><br>
 *
 * LinearRegressionSGD fits a model y = b<sub>0</sub> + b<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is using stochastic gradient descent.
 */
public class LinearRegressionSGD extends LinearRegression {

    protected double learningRate = 0.01;
    protected double threshold = 0.5e-5;
    protected int maxIterations = 5000;
    protected Optimizer GD;
    private final List<Double> lossHist = new ArrayList<>();

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
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate, int maxIterations, double threshold) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        validateParams();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        validateParams();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        validateParams();
    }


    /**
     *  Creates a {@link LinearRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(int maxIterations) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.maxIterations = maxIterations;
        validateParams();
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
        GD = new GradientDescent(learningRate);
        int[] shuffledIndices; // Stores shuffled indices for each epoch.

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Vector(features));
        Matrix y = new Vector(targets);

        Matrix wGrad;
        w = Matrix.randn(X.numCols(), 1, false); // initialize w.

        for(int i=0; i<maxIterations; i++) { // Apply stochastic gradient descent.
            shuffledIndices = ArrayUtils.randomIndices(X.numRows()); // Get randomly shuffled indices

            for(int j : shuffledIndices) { // Compute gradient a single sample at a time.
                wGrad = LinearGradient.getGrad(X.getRowAsVector(j), y.getRowAsVector(j), w); // Compute gradients
                w = GD.step(w, wGrad)[0]; // Apply gradient descent update rule.
            }

            // Append loss to the loss history.
            lossHist.add(LossFunctions.sse.compute(y, this.predict(X, w)).getAsDouble(0, 0));

            if(lossHist.get(lossHist.size()-1)<threshold) {
                break; // Then stop the training early
            }
        }

        // Update the model details
        super.coefficients = w.T().getValuesAsDouble()[0];
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

        return lossHist.stream().mapToDouble(Double::doubleValue).toArray();
    }


    private void validateParams() {
        if(maxIterations < 0)
            throw new IllegalArgumentException("Maximum iterations must be non-negative but got " + maxIterations + ".");
        if(learningRate < 0)
            throw new IllegalArgumentException("Learning rate must be non-negative but got " + learningRate + ".");
        if(threshold < 0)
            throw new IllegalArgumentException("Threshold must be non-negative but got " + threshold + ".");
    }
}
