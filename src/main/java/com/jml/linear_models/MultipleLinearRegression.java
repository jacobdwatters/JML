package com.jml.linear_models;

import com.jml.core.Model;
import java.util.ArrayList;
import java.util.Map;


/**
 * Model for least squares linear regression of multiple variables by stochastic gradient descent.<br><br>
 *
 * MultipleLinearRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x + ... + b<sub>n</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is using stochastic gradient descent.
 */
public class MultipleLinearRegression extends Model<ArrayList<double[][]>, double[]> {


    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile() {
        // TODO: Auto-generated method stub.
    }

    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile(Map<String, Double> args) {
        // TODO: Auto-generated method stub.
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
    public double[][] fit(ArrayList<double[][]> features, double[] targets, Map<String, Double> args) {
        // TODO: Auto-generated method stub.
        return new double[0][];
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
    public double[][] fit(ArrayList<double[][]> features, double[] targets) {
        // TODO: Auto-generated method stub.
        return new double[0][];
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
    public double[] predict(ArrayList<double[][]> features) {
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
}
