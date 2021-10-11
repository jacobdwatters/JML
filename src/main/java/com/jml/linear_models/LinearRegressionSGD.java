package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.core.ModelTypes;
import com.jml.losses.Function;
import com.jml.losses.LossFunctions;
import com.jml.losses.LossGradients;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.StochasticGradientDescent;
import linalg.Matrix;
import linalg.Vector;

import java.util.HashMap;
import java.util.Map;


/**
 * Model for least squares linear regression of one variable by stochastic gradient descent.<br><br>
 *
 * LinearRegressionSGD fits a model y = β<sub>0</sub> + β<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is using stochastic gradient descent.
 */
public class LinearRegressionSGD extends LinearRegression {

    private double learningRate = 0.002;
    private double threshold = 0.5e-5;
    private int maxIterations = 1000;
    private Optimizer SGD;


    /**
     * Creates a {@link LinearRegressionSGD} model. This will use a default learning rate of 0.002.
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
     * {@link com.jml.optimizers.StochasticGradientDescent Stochastic Gradient Descent}.
     */
    public LinearRegressionSGD(double learningRate, double threshold, int maxIterations) {
        super.MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.toString();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
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
    public ModelBucket fit(double[] features, double[] targets) {
        Map<String, Object> results = new HashMap<>();
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Vector(features));
        Matrix y = new Vector(targets);

        w = SGD.optimize(LossFunctions.sse, X, y);

        results.put("coefficients", w.T().getValuesAsDouble()[0]);

        // Update the model details
        super.isFit=true;
        buildDetails();

        return new ModelBucket(results);
    }



    public static void main(String[] args) {
        Model<double[], double[]> model = new LinearRegressionSGD(0.02, 200);

        double[] x = {0.05, 0.11, 0.15, 0.31, 0.46, 0.52, 0.7, 0.74, 0.82, 0.98, 1.171};
        double[] y = {0.956, 0.89, 0.832, 0.717, 0.571, 0.539, 0.378, 0.37, 0.306, 0.242, 0.104};

        model.compile();
        model.fit(x, y);

        System.out.println("\n\n" + model.getDetails() + "\n\n");
    }
}
