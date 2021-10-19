package com.jml.clasifiers;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.core.Stats;
import com.jml.core.Block;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import linalg.Matrix;
import linalg.Vector;


public class KNearestNeighbors extends Model<double[][], int[]> {

    String MODEL_TYPE = ModelTypes.K_NEAREST_NEIGHBORS.toString();
    protected int k, p;
    protected Matrix X; // Matrix representation of model features.
    protected Matrix y; // Matrix representation of model targets.
    protected boolean isFit = false;

    // Details of model in human-readable format.
    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No"
    );

    // TODO: Add checks for k and p > 0.

    /**
     * Creates a KNearestNeighbors model with a default k of 3. That is, the model will consider the three closest
     * neighbors when making class prediction. The default distance metric is euclidean distance.
     */
    public KNearestNeighbors() {
        this.k = 3;
        this.p = 2;
        buildDetails();
    }


    /**
     * Creates a KNearestNeighbors model with a specified k. The default distance metric is euclidean distance.
     * @param k Number of neighbors to consider when making a class prediction.
     */
    public KNearestNeighbors(int k) {
        this.k = k;
        this.p = 2;
        buildDetails();
    }


    /**
     * Creates a KNearestNeighbors model with a specified k and power parameter for distance metric.
     * @param k Number of neighbors to consider when making a class prediction.
     * @param p Power parameter of the Minkowski distance i.e. sum( | x<sub>i</sub> - y<sub>i</sub> | <sup>p</sup> ) <sup>1/p</sup>.
     *          If p=2, this is equivalent to the euclidean distance. If p=1 this is equivalent to the manhattan distance.
     */
    public KNearestNeighbors(int k, int p) {
        this.k = k;
        this.p = p;
        buildDetails();
    }


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
    // TODO: Construct a K-d tree so that the predict method does not need to use a brute force algorithm.
    public Model<double[][], int[]> fit(double[][] features, int[] targets) {
        if(features.length != targets.length) {
            throw new IllegalArgumentException("Number of features does not match the number of targets");
        }

        X = new Matrix(features);
        y = new Vector(targets, 1);
        isFit = true;
        buildDetails();

        return this;
    }


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     * @throws IllegalStateException    Thrown if the model has not been fit.
     */
    public int[] predict(double[][] features) {
        double[][] distances = new double[features.length][X.numRows()];
        int[] predictions = new int[features.length];
        int[] mindices;
        double[] classes;

        Matrix F = new Matrix(features);
        Matrix z;
        Matrix Xtakez; // Holds the result of X-z

        for(int i=0; i<features.length; i++) { // Compute distances
            z = F.getRowAsVector(i).extend(X.numRows());
            Xtakez = X.sub(z);

            for(int j=0; j<X.numRows(); j++) {
                distances[i][j] = Xtakez.getRowAsVector(j).norm(p).re;
            }

            mindices = Stats.minIndices(distances[i], this.k);
            classes = new double[mindices.length];

            for(int j=0; j<classes.length; j++) {
                classes[j] = (int) y.getAsDouble(0, mindices[j]);
            }

            predictions[i] = (int) Stats.mode(classes);
        }

        return predictions;
    }


    /**
     * Makes a prediction using a model by specifying the parameters of the model.
     * Unlike the other predict method, no model needs to be trained to use this method since the parameters provided
     * define a model.
     *
     * @param X Features to make prediction on.
     * @param F The parameters of the Model. In the case of KNN, there are no parameters and so this will be ignored.
     * @return prediction on the features using the given model parameters.
     */
    // TODO: F need to be [X|y](that is features and targets of the training set) and X needs to be the features to make a prediction on.
    @Override
    public Matrix predict(Matrix X, Matrix F) {
        return new Vector(predict(X.getValuesAsDouble()));
    }


    /**
     * Gets the parameters of the trained model.
     * Note that for a KNearestNeighbors model, there are no parameters to return.
     *
     * @return Features of model since there are no parameters in a KNearestNeighbors model. If model has not been fit,
     * returns null.
     */
    @Override
    public Matrix getParams() {
        return X;
    }


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    @Override
    public void saveModel(String filePath) {
        Block[] blockList;

        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }
        if(!filePath.endsWith(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        blockList = new Block[5];

        // Construct the blocks for the model file.
        blockList[0] = new Block(ClassifierTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(ClassifierTags.K.toString(), Integer.toString(this.k));
        blockList[2] = new Block(ClassifierTags.P.toString(), Integer.toString(this.p));
        blockList[3] = new Block(ClassifierTags.FEATURES.toString(), ArrayUtils.asString(X.getValuesAsDouble()));
        blockList[4] = new Block(ClassifierTags.CLASSES.toString(), ArrayUtils.asString(y.getValuesAsDouble()));

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }


    // Construct details of model
    protected void buildDetails() {
        details = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE+ "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n" +
                        "k-neighbors: " + k + "\n" +
                        "distance parameter: " + p
        );
    }


    /**
     * Forms a string of the important aspects of the model.<br>
     * same as {@link #toString()}
     *
     * @return Details of model as string.
     */
    @Override
    public String getDetails() {
        return details.toString();
    }


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    @Override
    public String toString() {
        return getDetails();
    }
}
