package com.jml.core;

import java.util.HashMap;


/**
 * This interface specified the requirements for a machine learning model.
 *
 * @param <E> The type of the model.
 * @param <X> The type of the features dataset.
 * @param <Y> The type of the targets dataset.
 */
public interface Models<E, X, Y> {

    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    void compile();


    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     * @param args A hashtable containing additional arguments in the form <name, value>.
     */
    void compile(HashMap<String, Double> args);


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
    double[][] fit(X features, Y targets, HashMap<String, Double> args);


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
    double[][] fit(X features, Y targets);


    /**
     * Uses fitted/trained model to make predictions on features.
     *
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     * the specification when the model was compiled.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     */
    Y predict(X features);


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    void saveModel(String filePath);


    /**
     * Prints details of model to the standard output.
     */
    void printDetails();
}
