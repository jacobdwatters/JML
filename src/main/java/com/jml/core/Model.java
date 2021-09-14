package com.jml.core;
import java.util.Map;


/**
 * This interface specified the requirements for a machine learning model.
 *
 * @param <X> The type of the features dataset.
 * @param <Y> The type of the targets dataset.
 */
public abstract class Model<X, Y> {

    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    public abstract void compile();


    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     * @param args A hashtable containing additional arguments in the form <name, value>.
     */
    public abstract void compile(Map<String, Double> args);


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     * - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     * - If the features and targets are not correctly sized per the specification when the model was
     * compiled.
     *
     * @param features The features of the training set.
     * @param targets The targets of the training set.
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @return Returns details of the fitting / training process.
     */
    public abstract double[][] fit(X features, Y targets, Map<String, Double> args);


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     * the specification when the model was compiled.
     *
     * @param features The features of the training set.
     * @param targets The targets of the training set.
     * @return - Returns details of the fitting / training process.
     */
    public abstract double[][] fit(X features, Y targets);


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     * the specification when the model was compiled.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     */
    public abstract Y predict(X features);


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    public abstract void saveModel(String filePath);


    /**
     * Forms a string of the important aspects of the model.<br>
     * same as {@link #toString()}
     *
     * @return Details of model as string.
     */
    public abstract String getDetails();


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    public abstract String toString();
}
