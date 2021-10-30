package com.jml.clasifiers;

import com.jml.core.Model;
import linalg.Matrix;
import linalg.Vector;

public class LogisticRegression extends Model<double[][], double[]> {

    protected Matrix w;

    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return This. i.e. the trained model.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public Model<double[][], double[]> fit(double[][] features, double[] targets) {
        // TODO: Auto-generated method stub.
        return null;
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
    public double[] predict(double[][] features) {
        Matrix predictions = new Vector(features.length);
        Matrix X = new Matrix(features);

        for(int i=0; i<X.numRows(); i++) {
            predictions.set(1 / (1+Math.pow(Math.E, -w.getColAsVector(0).innerProduct(X.getRowAsVector(i)).re)),
                    i, 0);
        }

        return predictions.T().getValuesAsDouble()[0];
    }


    /**
     * Makes a prediction using a model by specifying the parameters of the model.
     * Unlike the other predict method, no model needs to be trained to use this method since the parameters provided
     * define a model.
     *
     * @param w Parameters of the model
     * @param X Features to make prediction on
     * @return prediction on the features using the given model parameters.
     */
    @Override
    public Matrix predict(Matrix X, Matrix w) { // TODO: Currently only supports binary classification.

        Matrix predictions = new Vector(X.numRows());

        for(int i=0; i<X.numRows(); i++) {
            predictions.set(1 / (1+Math.pow(Math.E, -w.getColAsVector(0).innerProduct(X.getRowAsVector(i)).re)),
                    i, 0);
        }

        return predictions;
    }


    /**
     * Gets the parameters of the trained model.
     *
     * @return A matrix containing the parameters of the trained model.
     */
    @Override
    public Matrix getParams() {
        // TODO: Auto-generated method stub.
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
    public String getDetails() {
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
        return null;
    }
}
