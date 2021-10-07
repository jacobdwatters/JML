package com.jml.neural_network;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.neural_network.layers.Layer;
import linalg.Matrix;

import java.util.Map;

public class NeuralNetwork extends Model<double[][], double[][]> {


    private Layer<?, ?> layer;

    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile() {
        // TODO: Auto-generated method stub
    }

    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile(Map<String, Double> args) {
        // TODO: Auto-generated method stub
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
    public ModelBucket fit(double[][] features, double[][] targets, Map<String, Double> args) {
        // TODO: Auto-generated method stub
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
    public ModelBucket fit(double[][] features, double[][] targets) {
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * Uses fitted/trained model to make predictions on features.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public double[][] predict(double[][] features) {
        // TODO: Auto-generated method stub
        return new double[0][];
    }


    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Auto-generated method stub.
        return null;
    }


    /**
     * Adds specified layer to the network.
     *
     * @param layer Layer to add to the neural network.
     */
    public void add(Layer<?, ?> layer) {
        // TODO: Auto-generated method stub
    }


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    @Override
    public void saveModel(String filePath) {
        // TODO: Auto-generated method stub
    }


    /**
     * Forms a string of the important aspects of the model.<br>
     * same as {@link #toString()}
     *
     * @return Details of model as string.
     */
    @Override
    public String getDetails() {
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    @Override
    public String toString() {
        // TODO: Auto-generated method stub
        return "";
    }
}
