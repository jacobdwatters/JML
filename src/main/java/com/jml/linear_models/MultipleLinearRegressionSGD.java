package com.jml.linear_models;

import com.jml.core.Gradient;
import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.GradientDescent;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.Scheduler;
import com.jml.util.ValueError;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.List;


/**
 * Model for least squares linear regression of multiple variables by stochastic gradient descent.<br><br>
 *
 * MultipleLinearRegressionSGD fits a model y = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + ... + b<sub>n</sub>x<sub>n</sub>
 * to the datasets by minimizing the residuals of the sum of squares
 * (i.e. the {@link LossFunctions#sse sum of square errors}) between the values in the
 * target dataset and the values predicted by the model. This is minimized using stochastic gradient descent.
 */
public class MultipleLinearRegressionSGD extends MultipleLinearRegression {

    protected double learningRate = 0.002;
    protected double threshold = 0.5e-5;
    protected int maxIterations = 1000;
    private Optimizer GD;
    private List<Double> lossHist = new ArrayList<>();

    // TODO: Currently using standard gradient descent. Need to change to actual stochastic gradient descent.

    /**
     * Creates a {@link MultipleLinearRegressionSGD} model.  This will use a default learning rate of 0.002.
     */
    public MultipleLinearRegressionSGD() {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
    }



    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge within the threshold by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}
     * @param threshold Threshold for early stopping during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     *                  If the loss is less than the specified threshold, gradient descent will stop early.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(double learningRate, int maxIterations, double threshold) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        paramCheck();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(double learningRate, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        paramCheck();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param learningRate Learning rate to use during {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
     */
    public MultipleLinearRegressionSGD(double learningRate) {
        super.MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        paramCheck();
    }


    /**
     *  Creates a {@link MultipleLinearRegressionSGD} model. When the {@link #fit(double[][], double[]) fit}
     *  method is called, {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent} will use the
     *  provided learning rate and will stop if it does not converge by the specified number of max iterations.
     *
     * @param maxIterations Maximum number of iterations to run for during
     * {@link com.jml.optimizers.GradientDescent Stochastic Gradient Descent}.
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
        GD = new GradientDescent(learningRate);

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Matrix(features));
        Matrix y = new Vector(targets);

        Matrix wGrad;
        w = Matrix.randn(X.numCols(), 1, false); // initialize w.

        for(int i=0; i<maxIterations; i++) {
            for(int j=0; j<X.numRows(); j++) {
                // TODO: No need to numerically compute gradient here.
                wGrad = Gradient.compute(w, X.getRowAsVector(j), y.getRowAsVector(j), LossFunctions.sse, this); // Compute gradients
                w = GD.step(w, wGrad); // Apply gradient descent update rule.
            }

            // Append loss to the loss history.
            lossHist.add(LossFunctions.sse.compute(y, this.predict(X, w)).getAsDouble(0, 0));

            if(lossHist.get(lossHist.size()-1)<threshold) {
                break; // Then stop the training early
            }
        }

        super.coefficients = w.T().getValuesAsDouble()[0];
        super.isFit=true;
        super.buildDetails();

        return this;
    }


    /**
     * Gets the loss history from the optimizer.
     * @return Returns the loss for each iteration of the optimization algorithm in an array. The index of the array
     * corresponds to the iteration the loss was computed for.
     */
    public double[] getLossHist() {
        if(!isFit) {
            throw new IllegalStateException("Model must be trained before the loss history can be computed.");
        }

        return lossHist.stream().mapToDouble(Double::doubleValue).toArray();
    }


    // Ensure parameters are valid.
    private void paramCheck() {
        if(!ValueError.isNonNegative(maxIterations))
            throw new IllegalArgumentException("maxIterations must be non-negative but got " + maxIterations);
        if(!ValueError.isNonNegative(learningRate))
            throw new IllegalArgumentException("learningRate must be non-negative but got " + learningRate);
        if(!ValueError.isNonNegative(threshold))
            throw new IllegalArgumentException("threshold must be non-negative but got " + threshold);
    }
}
