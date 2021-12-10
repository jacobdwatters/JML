package com.jml.neural_network;

import com.jml.core.Block;
import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.neural_network.activations.ActivationFunction;
import com.jml.neural_network.layers.Layer;
import com.jml.util.FileManager;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.List;


/**
 * A class that supports the creation and training of neural networks. A neural network is a supervised
 * learning model that is structured in layers. <br><br>
 * Neural networks are created sequentially one layer at a time by using the {@link #add(Layer)} method.
 * Activation functions can be specified for applicable layers.
 *
 * <br><br>
 * Layers:
 * <pre>
 *     {@link com.jml.neural_network.layers.Dense Dense}
 *     {@link com.jml.neural_network.layers.Dropout Dropout}
 * </pre>
 * Activation Functions:
 * <pre>
 *     {@link com.jml.neural_network.activations.Activations#linear Linear}
 *     {@link com.jml.neural_network.activations.Activations#sigmoid Sigmoid}
 *     {@link com.jml.neural_network.activations.Activations#relu ReLU}
 * </pre>
 *
 * Custom layers or activation functions can be created using the
 * {@link com.jml.neural_network.activations.ActivationFunction ActivationFunction} and
 * {@link com.jml.neural_network.layers.Layer Layer} interfaces.
 */
public class NeuralNetwork extends Model<double[][], double[][]> {

    protected String MODEL_TYPE = ModelTypes.NEURAL_NETWORK.toString();
    protected final List<Layer> layers;
    protected double learningRate;
    protected double threshold;
    protected int epochs;
    protected int batchSize;

    protected boolean isFit = false;
    List<Double> lossHist = new ArrayList<>();

    private Matrix[] dxUpdates;
    private Matrix[] dxBiasUpdates;

    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );

    // TODO: Add other constructors.
    /**
     * Constructs a neural network with default hyper-parameters.
     * <pre>
     *     Learning Rate: 0.01
     *     epochs: 10
     *     batchSize: 1
     *     threshold: 1E-5
     * </pre>
     */
    public NeuralNetwork() {
        this.learningRate = 0.01;
        this.epochs = 10;
        this.batchSize = 1;
        this.threshold = 1e-5;
        layers = new ArrayList<>();

        buildDetails();
    }

    /**
     * Constructs a neural network with the specified hyper-parameters.
     *
     * @param learningRate The learning rate to be using during optimization.
     * @param epochs The number of epochs to train the network for.
     * @param batchSize The batch size to use during training.
     * @param threshold The threshold for the loss to stop early. If the loss drops below this threshold before the
     *                  specified number of epochs has been reached, the training will stop early.
     */
    // TODO: Will need to take optimizer as parameter. Take this as a string?
    public NeuralNetwork(double learningRate, int epochs, int batchSize, double threshold) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = threshold;
        layers = new ArrayList<>();

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
        dxUpdates = new Matrix[layers.size()]; // Weight updates
        dxBiasUpdates = new Matrix[layers.size()];

        resetDx();

        Matrix feature = new Matrix(features);
        Matrix target = new Matrix(targets);
        Matrix input;
        Matrix output;
        Matrix predictions;

        for(int i=0; i<epochs; i++) {
            for(int j=0; j<feature.numRows(); j++) {
                for(int k=0; k<batchSize && (j+k)<feature.numRows(); k++) {
                    input = feature.getRowAsVector(j+k).T();
                    output = feedForward(input);
                    back(target.getRowAsVector(j+k).T(), output, input);
                }

                applyUpdates();
            }

            predictions = new Matrix(this.predict(features));
            lossHist.add(LossFunctions.mse.compute(predictions, target).get(0, 0).re);

            if(lossHist.get(lossHist.size()-1) < threshold) {
                break; // Then stop training since the loss has dropped below the stopping threshold.
            }
        }

        isFit=true;
        buildDetails();

        return this;
    }


    /**
     * Computes the forward pass of the neural network.<br>
     * The forward pass computes the values for each layer based on an input.
     */
    protected Matrix feedForward(Matrix input) {
        Matrix currentInput = new Matrix(input);
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

        Matrix error = target.sub(output);
        ActivationFunction activation;

        // TODO: Update bias terms as well.
        if(layers.size()>1) {
            activation = layers.get(layers.size()-1).getActivation();

            dxUpdates[dxUpdates.length-1] = dxUpdates[dxUpdates.length-1]
                    .add(activation.slope(layers.get(layers.size()-1).getValues())
                    .elemMult(error)
                    .scalMult(learningRate)
                    .mult(layers.get(layers.size()-2).getValues().T()));

            dxBiasUpdates[dxBiasUpdates.length-1] = dxBiasUpdates[dxBiasUpdates.length-1]
                    .add(activation.slope(layers.get(layers.size()-1).getBias())
                    .elemMult(error)
                    .scalMult(learningRate));

            error = layers.get(layers.size()-1).getWeights().T().mult(error);

            for(int i=layers.size()-2; i>=1; i--) {
                activation = layers.get(i).getActivation();

                dxUpdates[i] = dxUpdates[i]
                        .add(activation.slope(layers.get(i).getValues())
                        .elemMult(error)
                        .scalMult(learningRate)
                        .mult(layers.get(i-1).getValues().T()));

                dxBiasUpdates[i] = dxBiasUpdates[i]
                        .add(activation.slope(layers.get(i).getBias())
                        .elemMult(error)
                        .scalMult(learningRate));

                error = layers.get(i).getWeights().T().mult(error);
            }

        }

        activation = layers.get(0).getActivation();
        dxUpdates[0] = dxUpdates[0]
                .add(activation.slope(layers.get(0).getValues())
                .elemMult(error)
                .scalMult(learningRate)
                .mult(input.T()));
        dxBiasUpdates[0] = dxBiasUpdates[0]
                .add(activation.slope(layers.get(0).getBias())
                .elemMult(error)
                .scalMult(learningRate));
    }


    /**
     * Applies weight updates to each layer.
     */
    private void applyUpdates() {
        for(int i=0; i<layers.size(); i++) { // Update the weights for each layer.
            layers.get(i).setWeights(dxUpdates[i].scalDiv(batchSize).add(layers.get(i).getWeights()));
            layers.get(i).setBias(dxBiasUpdates[i].scalDiv(batchSize).add(layers.get(i).getBias()));
        }

        resetDx(); // Reset dx's for next epoch.
    }

    private void resetDx() {
        for(int i=0; i<dxUpdates.length; i++) { // Initialize all weight updates to the zero matrix of appropriate size.
            dxUpdates[i] = new Matrix(layers.get(i).getWeights().shape());
            dxBiasUpdates[i] = new Vector(layers.get(i).getOutDim());
        }
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
     * Adds specified layer to the network.<br><br>
     *
     * The first layer must have a specified input dimension that matches
     * the number of columns of a feature array that will be used in the {@link #fit(double[][], double[][])} method. <br>
     * Subsequent layers may have no input dimension defined. In this case the input dimension will
     * be inferred from the output dimension of the previous layer.<br><br>
     *
     * The final layers output dimension must match the number of columns of the target array that is used in the
     * {@link #fit(double[][], double[][])} method.
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
        Block[] blockList;

        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }
        if(!filePath.endsWith(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        blockList = new Block[2 + layers.size()];

        StringBuilder hyperParams = new StringBuilder();
        hyperParams.append(this.learningRate + ", ");
        hyperParams.append(this.epochs + ", ");
        hyperParams.append(this.batchSize + ", ");
        hyperParams.append(this.threshold);

        // Construct the blocks for the model file.
        blockList[0] = new Block(ModelTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(ModelTags.HYPER_PARAMETERS.toString(), hyperParams.toString());

        int count = 2;
        StringBuilder layerDetails = new StringBuilder();
        for(Layer layer : layers) {
            blockList[count] = new Block(ModelTags.LAYER.toString(), layer.inspectTemp());
            count++;
        }

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
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
                details.append("\t" + layerCount + "\t" + layer.inspect() + "\n");
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
    public String inspect() {
        return this.details.toString();
    }


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    @Override
    public String toString() {
        return inspect();
    }
}
