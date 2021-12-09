package com.jml.core;

import com.jml.util.FileManager;
import linalg.Matrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/**
 * This interface specified the requirements for a machine learning model.
 *
 * @param <X> The type of the features dataset.
 * @param <Y> The type of the targets dataset.
 */
public abstract class Model<X, Y> {


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     * the specification when the model was compiled.
     *
     * @param features The features of the training set.
     * @param targets The targets of the training set.
     * @return This. i.e. the trained model.
     */
    public abstract Model<X, Y> fit(X features, Y targets);


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     * the specification when the model was compiled.
     * @throws IllegalStateException Thrown if the model has not been compiled and fit.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     */
    public abstract Y predict(X features);


    /**
     * Makes a prediction using a model by specifying the parameters of the model.
     * Unlike the other predict method, no model needs to be trained to use this method since the parameters provided
     * define a model.
     *
     * @param X Features to make prediction on.
     * @param w Parameters of the model.
     * @return prediction on the features using the given model parameters.
     */
    public abstract Matrix predict(Matrix X, Matrix w);


    /**
     * Gets the parameters of the trained model.
     *
     * @return A matrix containing the parameters of the trained model.
     */
    public abstract Matrix getParams();


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
    public abstract String inspect();


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

        if(!filePath.endsWith(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        String fileContent = FileManager.readFile(filePath);

        if(fileContent.equals("")) {
            return null;
        } else {
            List<String> lines = new ArrayList<>(),
                    blocks = new ArrayList<>();
            Collections.addAll(lines, fileContent.split("\n"));

            while(!lines.isEmpty()) {
                blocks.add(nextBlock(lines));
            }

            Class<Model> clazz = Model.class;

            return clazz.cast(ModelFromData.create(blocks));
        }
    }


    // TODO: Move this into the ModelFromData class.
    /**
     * A helper method which gets the next block from an ArrayList of file lines. This gets only outer-block.
     * That is, if a block has sub-blocks, the entire parent block, containing the sub-blocks, will be returned.
     * This also removes that block from the lines.
     *
     * @param lines Lines of file.
     * @return The string format of the block.
     */
    protected static String nextBlock(List<String> lines) {
        StringBuilder block = new StringBuilder();
        boolean foundStart = false, blockFound = false;
        String startTag = "", currentTag;

        while(!blockFound) {
            if(lines.get(0).matches("<(.*?)>") && lines.get(0).contains("<\\")) { // Then we have found an ending block
                if(!foundStart) {
                    throw new IllegalStateException("Unable to load model. Parser got stuck in a bad state.");
                }

                // Get the tag of the starting block. We want to continue
                currentTag = ModelFromData.getTag(lines.get(0).replace("\\", ""));

                block.append(lines.remove(0)); // Remove line and add it to block
                block.append("\n");

                if(startTag.equals(currentTag)) {
                    blockFound=true; // This is the end of the block
                }
            }
            else if(lines.get(0).matches("<(.*?)>") && !lines.get(0).contains("\\") && !foundStart) { // Then we have the beginning of a block
                foundStart = true;

                /*
                 * Get the tag of the starting block. We want to continue until we find a closing block with the same tag.
                 */
                startTag = ModelFromData.getTag(lines.get(0));

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
