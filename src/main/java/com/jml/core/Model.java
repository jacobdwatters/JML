package com.jml.core;

import com.jml.util.FileManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
     * @return Returns details of the fitting / training process in a {@link ModelBucket}. The
     * arguments passed may effect what the {@link ModelBucket}
     */
    public abstract ModelBucket fit(X features, Y targets, Map<String, Double> args);


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     * the specification when the model was compiled.
     *
     * @param features The features of the training set.
     * @param targets The targets of the training set.
     * @return Returns details of the fitting / training process in a {@link ModelBucket}.
     */
    public abstract ModelBucket fit(X features, Y targets);


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


    /**
     * Loads a model from specified file path including extension.<br>
     * Models must be saved as an MDL file (i.e. *.mdl).
     *
     * @param filePath File path, including file extension, of model to load.
     * @return Returns a saved trained model from the file path.
     */
    public static Model load(String filePath) {
        if(!filePath.substring(filePath.length()-4,filePath.length()).equals(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        Model model = null;
        String currentBlock;

        String fileContent = FileManager.readFile(filePath);
        List<String> lines = new ArrayList<String>(),
        blocks = new ArrayList<String>();
        Collections.addAll(lines, fileContent.split("\n"));

        while(!lines.isEmpty()) {
            blocks.add(nextBlock(lines));
        }

        Class<Model> clazz = Model.class;

        return clazz.cast(ModelFromData.create(blocks));
    }


    /**
     * A helper method which gets the next block from an ArrayList of file lines.
     * This also removes that block from the lines.
     *
     * @param lines Lines of file.
     * @return The string format of the block.
     */
    private static String nextBlock(List<String> lines) {
        StringBuilder block = new StringBuilder();
        boolean foundStart = false, blockFound = false;

        while(!blockFound) {
            if(lines.get(0).matches("<(.*?)>") && lines.get(0).contains("<\\")) {
                if(!foundStart) {
                    throw new IllegalStateException("Unable to load model. Parser got stuck in a bad state.");
                }

                block.append(lines.remove(0)); // Remove line and add it to block
                block.append("\n");

                blockFound=true; // This is the end of the block
            }
            else if(lines.get(0).matches("<(.*?)>") && !lines.get(0).contains("\\")) { // Then we have the beginning of a block
                foundStart = true;
                block.append(lines.remove(0)); // Remove line and add it to block
                block.append("\n");
            } else {
                block.append(lines.remove(0)); // Remove line and add it to block
                block.append("\n");
            }
        }

        return block.toString();
    }
}
