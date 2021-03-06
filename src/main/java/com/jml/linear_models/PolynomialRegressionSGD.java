package com.jml.linear_models;

import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.GradientDescent;
import com.jml.optimizers.Optimizer;
import com.jml.util.ArrayUtils;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.List;


/**
 * Model for least squares regression of polynomials using {@link GradientDescent Stochastic Gradient Descent}.<br><br>
 *
 * PolynomialRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x  + b<sub>2</sub>x<sup>2</sup> + ... +
 * b<sub>n</sub>x<sup>n</sup> to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is solved using Stochastic Gradient Descent.
 */
public class PolynomialRegressionSGD extends PolynomialRegression {
    protected double learningRate = 0.01;
    protected double threshold = 0.5e-5;
    protected int maxIterations = 5000;
    private Optimizer GD;
    private final List<Double> lossHist = new ArrayList<>();


    /**
     * Creates a {@link PolynomialRegressionSGD} model. This will use a default learning rate of 0.002.
     */
    public PolynomialRegressionSGD() {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.degree = 1;
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param degree Degree of the polynomial to fit.
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(int degree, double learningRate, int maxIterations, double threshold) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.degree = degree;
        validateParams();
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param degree Degree of the polynomial to fit.
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(int degree, double learningRate, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.degree = degree;
        validateParams();
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param degree Degree of the polynomial to fit.
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public PolynomialRegressionSGD(int degree, double learningRate) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.degree = degree;
        validateParams();
    }


    /**
     *  Creates a {@link PolynomialRegressionSGD} model. When the {@link #fit(double[], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will fit a
     *  polynomial of the specified degree using gradient descent.
     *
     * @param degree Degree of the polynomial to fit.
     */
    public PolynomialRegressionSGD(int degree) {
        super.MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION_SGD.toString();
        this.degree = degree;
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
    public PolynomialRegressionSGD fit(double[] features, double[] targets) {
        GD = new GradientDescent(learningRate);

        int[] shuffledIndices; // Stores shuffled indices for each epoch.

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.van( new Vector(features), degree+1);
        Matrix y = new Vector(targets);

        Matrix wGrad;
        w = Matrix.randn(X.numCols(), 1, false); // initialize w.

        for(int i=0; i<maxIterations; i++) { // Apply stochastic gradient descent.
            shuffledIndices = ArrayUtils.randomIndices(X.numRows()); // Get randomly shuffled indices

            for(int j : shuffledIndices) { // Compute gradient a single sample at a time.
                shuffledIndices = ArrayUtils.randomIndices(X.numRows()); // Get randomly shuffled indices
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


    // Ensure constructor parameters are valid.
    private void validateParams() {
        if(maxIterations<0)
            throw new IllegalArgumentException("Maximum iterations must be non-negative but got " + maxIterations + ".");
        if(learningRate<0)
            throw new IllegalArgumentException("Learning rate must be non-negative but got " + learningRate + ".");
        if(threshold<0)
            throw new IllegalArgumentException("Threshold must be non-negative but got " + threshold + ".");
        if(degree<1) {
            throw new IllegalArgumentException("Polynomial degree must be at least 1 but got " + degree + ".");
        }
    }
}
