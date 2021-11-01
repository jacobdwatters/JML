package com.jml.clasifiers;

import com.jml.core.Block;
import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.linear_models.LinearModelTags;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.Scheduler;
import com.jml.optimizers.StochasticGradientDescent;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import linalg.Matrix;
import linalg.Vector;

public class LogisticRegression extends Model<double[][], double[]> {

    protected boolean isFit = false;

    protected String MODEL_TYPE = ModelTypes.LOGISTIC_REGRESSION.toString();
    protected Matrix w;
    protected double[] coefficients;

    // Variables for optimization.
    protected double learningRate = 0.002;
    protected double threshold = 0.5e-5;
    protected int maxIterations = 1000;
    private Optimizer SGD;
    protected Scheduler schedule;

    // Details of model in human-readable format.
    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );


    public LogisticRegression() {
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);
    }


    /**
     * Creates a logistic regression model. The model will be {@link #fit(double[][], double[])} using a
     * {@link com.jml.optimizers.StochasticGradientDescent stochastic gradient descent} optimizer with specified
     * learning rate, max iterations, threshold, and schedule.
     *
     * @param learningRate Learning rate to use during optimization.
     * @param maxIterations Maximum iterations to run optimizer for.
     * @param threshold Threshold for stopping the optimizer. If the loss becomes less than this value, the optimizer
     *                  will stop early.
     * @param schedule learning rate scheduler for optimization.
     */
    public LogisticRegression(double learningRate, int maxIterations, double threshold, Scheduler schedule) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.schedule = schedule;
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);
        SGD.setScheduler(this.schedule);
    }


    public LogisticRegression(double learningRate, int maxIterations, double threshold) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);
    }


    public LogisticRegression(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);
    }


    public LogisticRegression(double learningRate) {
        this.learningRate = learningRate;
        SGD = new StochasticGradientDescent(this, learningRate, maxIterations, threshold);
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
    public Model<double[][], double[]> fit(double[][] features, double[] targets) {
        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Matrix(features));
        Matrix y = new Vector(targets);

        w = SGD.optimize(LossFunctions.binCrossEntropy, X, y); // Apply optimizer to the loss function
        this.coefficients = w.T().getValuesAsDouble()[0];

        isFit=true;
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
     * @throws IllegalStateException    Thrown if the model has not been compiled and fit.
     */
    @Override
    public double[] predict(double[][] features) {
        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }

        Matrix predictions = new Vector(features.length);
        Matrix X = Matrix.ones(features.length, 1).augment(new Matrix(features));

        for(int i=0; i<X.numRows(); i++) { // Apply the fitted logistic function to all features of X.
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
        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }

        return this.w;
    }


    /**
     * Gets the loss history from the optimizer.
     * @return Returns the loss for each iteration of the optimization algorithm in an array. The index of the array
     * corresponds to the iteration the loss was computed for.
     */
    public double[] getLossHist() {
        if(!isFit) {
            throw new IllegalStateException("Model must be trained before the loss history can be computed.");
        }

        return SGD.getLossHist().stream().mapToDouble(Double::doubleValue).toArray();
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

        blockList = new Block[2];

        // Construct the blocks for the model file.
        blockList[0] = new Block(LinearModelTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(LinearModelTags.PARAMETERS.toString(), ArrayUtils.asString(this.coefficients));

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }


    public void buildDetails() {
        details = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE + "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );

        if(isFit && coefficients!=null) {
            details.append("Coefficients: ");
            details.append(ArrayUtils.asString(coefficients));
            details.append("\nlogistic curve: y = 1 / [1+e^-{" + coefficients[0] + " + ");

            for(int i=1; i<coefficients.length; i++) {
                details.append(coefficients[i] + "*x_" + i);

                if(i<coefficients.length-1) {
                    details.append(" + ");
                }
            }

            details.append("}]");
        }
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


    public static void main(String[] args) {
        double[][] features = {{4}, {5}, {6}, {7}, {-1}, {0}, {2}};
        double[] targets = {1, 1, 1, 1, 0, 0, 0};

        double[][] test = {{3}, {3.5}, {4.8}};

        LogisticRegression logreg = new LogisticRegression(0.8, 10000);
        logreg.fit(features, targets);

        System.out.println("\n\n" + logreg.getDetails() + "\n");
        System.out.println("\n\n" + ArrayUtils.asString(logreg.predict(test)) + "\n");
        logreg.saveModel("temp.mdl");

        Model<double[][], double[]> logFromFile = Model.load("temp.mdl");
        System.out.println("\n\n" + logFromFile.getDetails() + "\n");
        System.out.println("\n\n" + ArrayUtils.asString(logFromFile.predict(test)) + "\n");
    }
}
