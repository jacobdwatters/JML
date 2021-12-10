package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.neural_network.NeuralNetwork;
import com.jml.neural_network.activations.ActivationFunction;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Layer;

import linalg.Matrix;


/**
 * A perceptron is a linear model that is equivalent to a single layer neural network.<br><br>
 *
 * When a perceptron model is saved, it will be saved as a neural network model.
 *
 * When the activation of a perceptron is the sigmoid function, it is a linear classifier that is analogous to
 * logistic regression.
 */
public class Perceptron extends Model<double[][], double[][]> {

    final String MODEL_TYPE = ModelTypes.PERCEPTRON.toString();
    NeuralNetwork perceptron;
    ActivationFunction activation;
    Layer layer;

    double learningRate, threshold;
    int epochs, batchSize;
    boolean isFit = false;

    StringBuilder inspection = new StringBuilder(
            "Model Details\n" +
            "----------------------------\n" +
            "Model Type: " + this.MODEL_TYPE+ "\n" +
            "Is Trained: No\n"
    );


    /**
     * Creates a perceptron with default hyper-parameters. The default parameters are listed below.
     * <pre>
     *     Learning Rate: 0.01
     *     Epochs: 10
     *     Batch Size: 1
     *     Threshold: 1e-5
     *     Activation: Sigmoid Function
     * </pre>
     */
    public Perceptron() {
        this.learningRate = 0.01;
        this.epochs = 10;
        this.batchSize = 1;
        this.threshold = 1e-5;
        this.activation = Activations.sigmoid;

        this.perceptron = new NeuralNetwork(this.learningRate, this.epochs, this.batchSize, this.threshold);

        buildDetails();
    }


    /**
     * Creates a perceptron with specified hyper-parameters and the sigmoid activation function. <br>
     * To specify an activation function, see {@link #Perceptron(double, int, int, double, ActivationFunction)}.
     *
     * @param learningRate Learning rate to use during training.
     * @param epochs Number of epochs to train the perceptron for.
     * @param batchSize Batch size to use during training.
     * @param threshold Threshold for early stopping of training. If the loss of the
     *                  perceptron model falls below this value during training, training will end before the specified
     *                  number of epochs.
     */
    public Perceptron(double learningRate, int epochs, int batchSize, double threshold) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = threshold;
        this.activation = Activations.sigmoid;

        this.perceptron = new NeuralNetwork(this.learningRate, this.epochs, this.batchSize, this.threshold);
        buildDetails();
    }


    /**
     * Creates a perceptron with specified hyper-parameters and activation function.
     *
     * @param learningRate Learning rate to use during training.
     * @param epochs Number of epochs to train the perceptron for.
     * @param batchSize Batch size to use during training.
     * @param threshold Threshold for early stopping of training. If the loss of the
     *                  perceptron model falls below this value during training, training will end before the specified
     *                  number of epochs.
     * @param activation Activation function to use in the perceptron.
     */
    public Perceptron(double learningRate, int epochs, int batchSize, double threshold, ActivationFunction activation) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = threshold;
        this.activation = activation;

        this.perceptron = new NeuralNetwork(this.learningRate, this.epochs, this.batchSize, this.threshold);
        buildDetails();
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Perceptron fit(double[][] features, double[][] targets) {
        if(isFit) {
           throw new IllegalStateException("Model has already been fit. Can not fit again.");
        }
        if(targets[0].length != 1) {
            throw new IllegalArgumentException("Perceptron can only have output dimension of 1 but got targets with " +
                    "dimension of " + targets[0].length + ". Target shape must be (n, 1) for n training samples.");
        }
        if(targets.length != features.length) {
            throw new IllegalArgumentException("Features and targets do not have the same number of samples. Got " +
                    features.length + " and " + targets.length + ".");
        }

        layer = new Dense(features[0].length, 1, activation);
        perceptron.add(layer); // Add layer to the perceptron.
        perceptron.fit(features, targets);
        this.isFit = true;
        buildDetails();

        return this;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public double[][] predict(double[][] features) {
        return perceptron.predict(features);
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Unneeded in perceptron.
        return null;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix getParams() {
        return null;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void saveModel(String filePath) {
        perceptron.saveModel(filePath);
    }


    /**
     * Builds the details of the model as a string representation of the important aspects of the model.
     */
    protected void buildDetails() {
        inspection = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE+ "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );

        inspection.append("Learning Rate: " + this.learningRate + "\n");
        inspection.append("Batch Size: " + this.batchSize + "\n");

        if(layer != null) {
            inspection.append("Layer:\n" + "------------\n");
            inspection.append("\t" + layer.inspect());
        }
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public String inspect() {
        return this.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return inspection.toString();
    }
}
