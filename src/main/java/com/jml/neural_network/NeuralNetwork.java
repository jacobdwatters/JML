package com.jml.neural_network;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Layer;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.StochasticGradientDescent;
import linalg.Matrix;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork extends Model<double[][], double[][]> {

    protected String MODEL_TYPE = ModelTypes.NEURAL_NETWORK.toString();
    private List<Layer> layers;
    protected double learningRate;
    protected double threshold;
    protected int epochs;
    protected int batchSize;
    protected Optimizer optim;


    /**
     * Constructs a neural network with the specified hyper-parameters.
     *
     * @param learningRate The learning rate to be using during optimization.
     * @param epochs The number of epochs to train the network for.
     * @param batchSize The batch size to use during training.
     * @param threshold The threshold for the loss to stop early. If the loss drops below this threshold before the
     *                  specified number of epochs has been reached, the training will stop early.
     */
    public NeuralNetwork(double learningRate, int epochs, int batchSize, double threshold) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = threshold;
        layers = new ArrayList<Layer>();
    }

    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return Returns details of the fitting / training process.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    @Override
    public NeuralNetwork fit(double[][] features, double[][] targets) {
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * Computes the forward pass of the neural network.<br>
     * The forward pass computes the values for each layer based on an input.
     */
    protected Matrix feedForward(Matrix inputs) {
        Matrix currentInput = new Matrix(inputs);
        for(Layer layer : layers) { // Feeds the input through all layers.
            currentInput = layer.forward(currentInput);
        }

        return currentInput;
    }


    /**
     * Computes the backward pass of the neural network and updates weights.
     * The backwards pass updates the weights of each layer in an attempt to decrease the loss of the forward pass.
     * This is done by back-propagation and gradient descent.
     */
    protected void back() {
        // TODO: Auto-generated method stub
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


    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Auto-generated method stub.
        return null;
    }

    /**
     * Gets the parameters of the trained model.
     *
     * @return A matrix containing the parameters of the trained model.
     */
    @Override
    public Matrix getParams() {
        // TODO: Implementation: will need a 'public Matrix[] getParams()' or 'public Matrix getParams(int layerIndex)'
        return null;
    }


    /**
     * Adds specified layer to the network.
     *
     * @param layer Layer to add to the neural network.
     */
    public void add(Layer layer) {
        if(layers.size() == 0) { // Then this is the first layer and the input dimension must be defined
            if(layer.getInDim() == -1) {
                throw new IllegalArgumentException("First layer must have input dimension defined.");
            }

        } else { // Then this is not the first layer.
            if(layer.getInDim() == -1) { // Then the input dimension is to be inferred from the previous layer.
                layer.updateInDim(layers.get(layers.size()-1).getOutDim()); // Infer the input dimension from previous layer

            } else {
                if(layer.getInDim() != layers.get(layers.size()-1).getOutDim()) {
                    throw new IllegalArgumentException("Layers input dimension of " + layer.getInDim() +
                            "\nis inconsistent with the previous layers output dimension of " + layers.get(layers.size()-1).getOutDim() + "." +
                            "\nLayers input dimension must match the output dimension of the previous layer.");
                }
            }
        }


        layers.add(layer);
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


    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(0.02, 100, 1, 0.5e-5);

        Matrix input = new Matrix(new double[][]{{0}, {1}});

        nn.add(new Dense(2, 3, Activations.sigmoid));
        nn.add(new Dense(1, Activations.sigmoid));

        System.out.println(nn.feedForward(input));
    }
}
