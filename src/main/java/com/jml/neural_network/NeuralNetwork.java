package com.jml.neural_network;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.neural_network.activations.Activation;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Dropout;
import com.jml.neural_network.layers.Layer;
import com.jml.optimizers.Optimizer;
import com.jml.optimizers.StochasticGradientDescent;
import com.jml.util.ArrayUtils;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork extends Model<double[][], double[][]> {

    protected String MODEL_TYPE = ModelTypes.NEURAL_NETWORK.toString();
    private List<Layer> layers;
    protected double learningRate;
    protected double threshold;
    protected int epochs;
    protected int batchSize;
    protected Optimizer optim;

    protected boolean isFit = false;

    private Matrix[] dxUpdates;

    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );


    /**
     * Constructs a neural network with the specified hyper-parameters.
     *
     * @param learningRate The learning rate to be using during optimization.
     * @param epochs The number of epochs to train the network for.
     * @param batchSize The batch size to use during training.
     * @param threshold The threshold for the loss to stop early. If the loss drops below this threshold before the
     *                  specified number of epochs has been reached, the training will stop early.
     */
    // TODO: Will ned to take optimizer as parameter. Take this as a string?
    public NeuralNetwork(double learningRate, int epochs, int batchSize, double threshold) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = threshold;
        layers = new ArrayList<Layer>();

        buildDetails();
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
        dxUpdates = new Matrix[layers.size()];

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
     *
     * @param target Target for the neural network. i.e. the expected or desired output for the given input sample.
     * @param output Output of the neural network. i.e. the result of the feed forward operation.
     */
    protected void back(Matrix target, Matrix output, Matrix input) {
        // TODO: Auto-generated method stub

        /*
        double[][] error = Matrix.subtract(currentTarget, outputValues);	// Error of output layer
		double[][] hiddenError;

        // Gradient Descent
        deltaHidden2OutputWeights =	Matrix.add(deltaHidden2OutputWeights,
                Matrix.multiply(
                        Matrix.scalMultiply(
                                Matrix.elementMultiply(
                                        sigmoidSlope(outputValues), error
                                ),
                                learningRate
                        ),
                        Matrix.transpose(hiddenValues[hiddenValues.length-1])
                )
         );

         hiddenError = Matrix.multiply(Matrix.transpose(hidden2OutputWeights), error);	// Error of hidden layer

         for(int i = deltaHiddenWeights.length-1; i >= 0; i--) {
            deltaHiddenWeights[i] =	Matrix.add(deltaHiddenWeights[i],
                    Matrix.multiply(
                            Matrix.scalMultiply(
                                    Matrix.elementMultiply(
                                            sigmoidSlope(hiddenValues[i+1]), hiddenError
                                    ),
                                    learningRate
                            ),
                            Matrix.transpose(hiddenValues[i])
                    )
                );

             hiddenError = Matrix.multiply(Matrix.transpose(hiddenWeights[hiddenWeights.length-1-i]), hiddenError);
          }

          deltaInput2HiddenWeights = Matrix.add(deltaInput2HiddenWeights,
					Matrix.multiply(
							Matrix.scalMultiply(
									Matrix.elementMultiply(
											sigmoidSlope(hiddenValues[0]), hiddenError
									),
									learningRate
							),
							Matrix.transpose(inputValues)
					)
				);
         */

        Activation sigDx = Activations.sigmoidSlope;
        Matrix dx; // Jacobian matrix. i.e. the matrix of partial derivatives.
        Layer layer;
        Matrix error = target.sub(output);

        // TODO: need separate dx for each layer.
        dx = new Matrix(layers.get(layers.size()-1).getWeights().shape()); // TODO: this is temp
        dx = dx.add(sigDx.apply(layers.get(layers.size()-1).getValues())
                .elemMult(error)
                .scalMult(learningRate)
                .mult(layers.get(layers.size()-2).getValues().T()));

        error = layers.get(layers.size()-1).getWeights().T().mult(error);

        for(int i=layers.size()-2; i>=1; i--) {
            dx = new Matrix(layers.get(layers.size()-1).getWeights().shape()); // TODO: this is temp
            dx = dx.add(sigDx.apply(layers.get(i).getValues())
                    .elemMult(error)
                    .scalMult(learningRate)
                    .mult(layers.get(i-1).getValues().T()));

            error = layers.get(i).getWeights().T().mult(error);
        }

        dx = new Matrix(layers.get(0).getWeights().shape()); // TODO: this is temp
        dx = dx.add(sigDx.apply(layers.get(0).getValues())
                .elemMult(error)
                .scalMult(learningRate)
                .mult(input.T()));

        // TODO: Input
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
        double[][] results = new double[features.length][layers.get(layers.size()-1).getOutDim()];

        for(int i=0; i<features.length; i++) {
            results[i] = feedForward(new Vector(features[i])).T().getValuesAsDouble()[0];
        }

        return results;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Auto-generated method stub.
        // TODO: This would have to be a prediction for a specific layer. Does this make sense to be here?
        // Answer, probably not
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
        buildDetails();
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


    protected void buildDetails() {
        details = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE+ "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );

        details.append("Learning Rate: " + this.learningRate + "\n");
        details.append("Batch Size: " + this.batchSize + "\n");

        if(!layers.isEmpty()) {
            details.append("Layers (" + layers.size() + "):\n" + "------------\n");

            int layerCount = 1;
            for(Layer layer : this.layers) {
                details.append("\t" + layerCount + "\t" + layer.getDetails() + "\n");
                layerCount++;
            }
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
        return this.details.toString();
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

        double[][] X = {{0, 1},
                        {1, 2}};

        NeuralNetwork nn = new NeuralNetwork(0.02, 100, 1, 0.5e-5);
        nn.add(new Dense(2, 3, Activations.sigmoid));
        nn.add(new Dense(1, Activations.sigmoid));
        System.out.println(nn.getDetails());

        Matrix target = new Matrix(new double[][]{{1}});
        Matrix input = new Matrix(new double[][]{{0}, {1}});
        Matrix output = nn.feedForward(input);
        nn.back(target, output, input);
    }
}
