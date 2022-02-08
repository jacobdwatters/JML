package com.jml.classifiers.naive_bayes;


import com.jml.core.Model;
import linalg.Matrix;

/**
 * A Bernoulli Naive Bayes model. Fits a dataset by assuming the dataset if drawn from a Bernoulli distribution
 * and that each feature has zero covariance. Then classification predictions can be made based on these assumptions.
 */
public class BernoulliNaiveBayes extends Model<int[][], int[]> {

    private double p; // Probability of success.

    /**
     * Fits or trains the model with the given features and targets. For both the features and targets parameters,
     * if they are 2D arrays, then the number of rows in each must match and will be the number of samples in the
     * data. The number of columns in each will be the number of features and targets in a single sample.<br><br>
     * <p>
     * For instance, if the features array has shape n-by-m and the targets array has shape n-by-k. Then there
     * are n samples in the dataset, each individual sample has m features, and each individual sample has k targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return This. i.e. the trained model.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public Model<int[][], int[]> fit(int[][] features, int[] targets) {
        // TODO: Auto-generated method stub.
        return this;
    }



    // TODO: Does p need to be an array?
    private static double[] bernoulliPdf(int[] x, double p) {
        double[] probabilities = new double[x.length];

        for(int i=0; i<x.length; i++) {
            if(x[i]==0) {
                probabilities[i] = 1-p;
            } else {
                probabilities[i] = p;
            }
        }

        return probabilities;
    }


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     * @throws IllegalStateException    Thrown if the model has not been compiled and fit.
     */
    @Override
    public int[] predict(int[][] features) {
        // TODO: Auto-generated method stub.
        return new int[0];
    }


    /**
     * Makes a prediction using a model by specifying the parameters of the model.
     * Unlike the other predict method, no model needs to be trained to use this method since the parameters provided
     * define a model.
     *
     * @param X Features to make prediction on.
     * @param w Parameters of the model.
     * @return prediction on the features using the given model parameters.
     */
    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Does nothing for this model. Need to remove requirement from Model class.
        //  Models that need this can inherit from a 'ParameterizedModel' abstract class.
        return null;
    }


    /**
     * Gets the parameters of the trained model.
     *
     * @return A matrix containing the parameters of the trained model.
     */
    @Override
    public Matrix getParams() {
        // TODO: Does nothing for this model. Need to remove requirement from Model class.
        //  Models that need this can inherit from a 'ParameterizedModel' abstract class.
        return null;
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
    public String inspect() {
        // TODO: Auto-generated method stub.
        return null;
    }


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    @Override
    public String toString() {
        // TODO: Auto-generated method stub.
        return null;
    }
}
