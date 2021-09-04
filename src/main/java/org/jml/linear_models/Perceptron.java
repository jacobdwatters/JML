package org.jml.linear_models;

import org.jml.core.Models;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;

public class Perceptron implements Models<Perceptron, Object, Object> {
    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile(HashMap<String, Double> args) {
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
    public double[][] fit(Object features, Object targets, HashMap<String, Double> args) {

        // TODO: This is just temporary for testing --------------------------------------------------------------------
        if(features.getClass().isArray()) {
            System.out.println("We have an array!");
        } else {
            System.out.println("We don't have an array :(");
        }

        // This gets the number of dimensions the array has. This would go in a utility class.
        Object o = features;
        Class<?> c = features.getClass();
        ArrayList<Integer> dimensionSizes = new ArrayList<>();
        int dimensions = 0;
        while (c.isArray()) {
            dimensions++;
            dimensionSizes.add(Array.getLength(o));
            o = Array.get(o, 0);
            if (o == null) {
                break;
            }
            c = o.getClass();
        }

        System.out.println("Dimensions: " + dimensions);
        System.out.print("Sizes: ");

        for(int i : dimensionSizes) {
            System.out.print(i + ",");
        }
        // END OF TESTING ----------------------------------------------------------------------------------------------

        // TODO: Auto-generated method stub
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
    public double[][] fit(Object features, Object targets) {
        // TODO: Auto-generated method stub
        return new double[0][];
    }

    /**
     * Uses fitted/trained model to make predictions on features.
     *
     * @param features
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public Object predict(Object features) {
        // TODO: Auto-generated method stub
        return null;
    }

    /**
     * Loads a trained model from a specified file containing a fitted / trained model.
     *
     * @param filePath File path, including extension, of fitted / trained model to be loaded.
     * @return The fitted / trained model located in the specified file.
     */
    @Override
    public Perceptron loadModel(String filePath) {
        // TODO: Auto-generated method stub
        return null;
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
     * Prints details of model to the standard output.
     */
    @Override
    public void printDetails() {
        // TODO: Auto-generated method stub
    }
}
