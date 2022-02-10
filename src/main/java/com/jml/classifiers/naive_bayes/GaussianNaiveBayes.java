package com.jml.classifiers.naive_bayes;

import com.jml.core.Block;
import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.core.Stats;
import com.jml.linear_models.LinearModelTags;
import com.jml.preprocessing.DataSplitter;

import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import linalg.Matrix;

import java.util.*;


/**
 * A Gaussian Naive Bayes model. Fits a dataset by assuming the dataset if drawn from a Gaussian / normal distribution
 * and that each feature has zero covariance. Then classification predictions can be made based on these assumptions.
 */
public class GaussianNaiveBayes extends Model<double[][], double[]> {
    // TODO: All classifiers should have a 'predictProbabilities()' Method.
    protected final String MODEL_TYPE = ModelTypes.GAUSSIAN_NAIVE_BAYES.toString();
    protected boolean isFit = false;

    private Map<Integer, List<double[]>> data;
    private Map<Integer, double[]> meansByFeature = new HashMap<>(); // Mean for each column of the features for each class.
    private Map<Integer, double[]> stdsByFeature = new HashMap<>(); // Standard deviation for each column of the features for each class.
    private Map<Integer, Double> priors = new HashMap<>();

    private double[][] features;
    private double[] targets;

    private StringBuilder inspection = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );


    /**
     * Creates a Gaussian Naive Bayes classification model.
     */
    public GaussianNaiveBayes() {
        // Does nothing.
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return This. i.e. the trained model.
     */
    @Override
    public GaussianNaiveBayes fit(double[][] features, double[] targets) {
        this.features = features;
        this.targets = targets;
        data = DataSplitter.splitByClass(features, ArrayUtils.toInt(targets));
        summarize(); // Summarize the data.
        this.isFit = true;

        buildInspection(); // build the inspection of this model.

        return this;
    }


    // Computes the mean and standard deviation for each feature for each class.
    private void summarize() {
        double[][] classData; // Data for a single class
        double[] means;
        double[] stds;

        for(int klass : data.keySet()) {
            classData = new Matrix(ArrayUtils.toDouble2D(data.get(klass).toArray())).T().getValuesAsDouble();
            means = new double[classData.length];
            stds = new double[classData.length];

            for(int i=0; i<classData.length; i++) {
                means[i] = Stats.mean(classData[i]);
                stds[i] = Stats.std(classData[i]);
            }

            // Insert the mean and standard deviation for each feature for this class.
            meansByFeature.put(klass, means);
            stdsByFeature.put(klass, stds);
            priors.put(klass, Math.log((double) classData[0].length/this.features.length));
        }
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
        if(features[0].length != this.features[0].length) {
            throw new IllegalArgumentException("Can not make predictions on data with " + features[0].length +
                    " columns. Expecting " + this.features[0].length + ".");
        }
        if (!isFit) {
            throw new IllegalStateException("Model must be fit before predictions can be made.");
        }

        double[] predictions = new double[features.length];
        double[] means;
        double[] stds;

        double post;
        double prior;
        Map<Integer, Double> posteriors = new HashMap<>();


        for(int i=0; i< features.length; i++) { // Iterate over all the samples
            for(int klass : meansByFeature.keySet()) {
                means = meansByFeature.get(klass);
                stds = stdsByFeature.get(klass);
                prior = priors.get(klass);

                post = Stats.sum(log(normalPdf(features[i], means, stds))) + prior;
                posteriors.put(klass, post);
            }

            predictions[i] = argmax(posteriors);
        }

        return predictions;
    }

    // TODO: add predictProbabilities(...) for all classification models. The predict(...) method should return the class prediction
    //  while the predictProbabilities(...) method will return a vector will the confidence for each class as a probability
    //  (i.e. the entries of the vector sum to 1.0).

    /**
     * Computes the value of the normal/gaussian distribution probability density function with specified mean and
     * standard deviation.
     *
     * @param x Value to evaluate probability density function at.
     * @param mean Mean of the normal/gaussian distribution.
     * @param std Standard deviation of the normal/gaussian distribution.
     * @return normal/gaussian distribution probability density function with specified mean and standard deviation
     * evaluated at x.
     */
    private static double[] normalPdf(double[] x, double[] mean, double[] std) {
        if(x.length != mean.length || x.length != std.length) {
            throw new IllegalArgumentException("All arrays should have the same length but got " + x.length + "," +
                    mean.length + ", " + std.length + ".");
        }

        double[] probabilities = new double[x.length];

        for(int i=0; i<x.length; i++) {
            // Compute gaussian probability for each feature and its respective mean and standard deviation.
            probabilities[i] = (1/(std[i]*Math.sqrt(2*Math.PI))) *
                    Math.exp(- Math.pow(x[i]-mean[i], 2) / (2*std[i]*std[i]));
        }

        return probabilities;
    }


    /**
     * Applies natural log transform, element-wise, to an array.
     * @param arr Array with values to apply transform to you.
     * @return An array containing the natural log of the elements of arr.
     */
    private static double[] log(double[] arr) {
        double[] logArr = new double[arr.length];

        for(int i=0; i<arr.length; i++) {
            logArr[i] = Math.log(arr[i]);
        }

        return logArr;
    }


    /**
     * Applies the argmax(x) function on a map x. That is, finds the key with the largest value.
     *
     * @param map Map of values.
     * @return The key which corresponds to the largest value in the map.
     */
    private static int argmax(Map<Integer, Double> map) {
        int maxKey = 0;
        double currMax = -Double.MAX_VALUE; // Holds the current maximum value.

        for(int key : map.keySet()) {
            if(map.get(key) > currMax) {
                currMax = map.get(key);
                maxKey = key;
            }
        }

        return maxKey;
    }


    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: This method is not needed for this model. Should not be required by model class...
        return null;
    }


    @Override
    public Matrix getParams() {
        // TODO: This method is not needed for this model. Should not be required by model class...
        return null;
    }


    @Override
    public void saveModel(String filePath) {
        Block[] blockList;

        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }
        if(!filePath.endsWith(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        blockList = new Block[3];

        // Construct the blocks for the model file.
        blockList[0] = new Block(LinearModelTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(LinearModelTags.FEATURES.toString(), ArrayUtils.asString(this.features));
        blockList[2] = new Block(LinearModelTags.TARGETS.toString(), ArrayUtils.asString(this.targets));

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }



    protected void buildInspection() {
        inspection = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE + "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );

        inspection.append("Input size: " + features[0].length);
    }


    @Override
    public String inspect() {
        return inspection.toString();
    }


    @Override
    public String toString() {
        // TODO: Auto-generated method stub.
        return null;
    }
}
