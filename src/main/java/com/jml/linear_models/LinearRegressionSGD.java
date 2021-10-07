package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.core.ModelTypes;
import com.jml.losses.Function;
import com.jml.losses.LossGradients;
import com.jml.optimizers.StochasticGradientDescent;
import linalg.Matrix;
import linalg.Vector;

import java.util.Map;


/**
 * Model for least squares linear regression of one variable by stochastic gradient descent.<br><br>
 *
 * LinearRegressionSGD fits a model y = b<sub>0</sub> + b<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is using stochastic gradient descent.
 */
public class LinearRegressionSGD extends Model<double[], double[]> {
    final String MODEL_TYPE = ModelTypes.LINEAR_REGRESSION_SGD.name();
    protected Matrix w;


    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile() {
        compile(null);
    }

    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile(Map<String, Double> args) {

    }

    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @param args     A hashtable containing additional arguments in the form <name, value>.
     * @return Returns details of the fitting / training process.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    @Override
    public ModelBucket fit(double[] features, double[] targets, Map<String, Double> args) {

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Vector(features));
        Matrix y = new Vector(targets);
        // Set initial model weights to be random.
        w = Matrix.randn(X.numCols(), 1, false);

        w = StochasticGradientDescent.optimize(w, X, y, LossGradients.sseLinRegGrad, 0.02, 2000);

        System.out.println("\n\nw:\n" + w + "\n\n");

        return null;
    }

    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return - Returns details of the fitting / training process.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public ModelBucket fit(double[] features, double[] targets) {
        return fit(features, targets, null);
    }


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public double[] predict(double[] features) {
        // TODO: Auto-generated method stub.
        return new double[0];
    }


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    @Override
    public void saveModel(String filePath) {
        // TODO: Auto-generated method stub.
    }


    /**
     * Forms a string of the important aspects of the model.<br>
     * same as {@link #toString()}
     *
     * @return Details of model as string.
     */
    @Override
    public String getDetails() {
        // TODO: Auto-generated method stub.
        return this.toString();
    }


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    @Override
    public String toString() {
        // TODO: Auto-generated method stub.
        return "";
    }


    public static void main(String[] args) {
        Model<double[], double[]> model = new LinearRegressionSGD();

        double[] x = {0.05, 0.11, 0.15, 0.31, 0.46, 0.52, 0.7, 0.74, 0.82, 0.98, 1.171};
        double[] y = {0.956, 0.89, 0.832, 0.717, 0.571, 0.539, 0.378, 0.37, 0.306, 0.242, 0.104};

        model.compile();
        model.fit(x, y);
    }
}
