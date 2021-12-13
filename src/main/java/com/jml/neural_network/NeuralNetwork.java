package com.jml.neural_network;

import com.jml.core.*;
import com.jml.losses.LossFunctions;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Dropout;
import com.jml.neural_network.layers.Layer;
import com.jml.optimizers.GradientDescent;
import com.jml.optimizers.Momentum;
import com.jml.optimizers.Optimizer;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.Arrays;
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

    private int trainableLayers=0;

    protected boolean isFit = false;
    List<Double> lossHist = new ArrayList<>();

    private Matrix[] dxUpdates;
    private Matrix[] dxBiasUpdates;
    private Matrix[] V; // Momentum update matrices. Only used for the Momentum optimizer.

    private Optimizer optim; // Optimizer to use during backpropagation.

    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );


    /**
     * Constructs a neural network with default hyper-parameters.
     * <pre>
     *     Learning Rate: 0.01
     *     epochs: 10
     *     batchSize: 1
     *     threshold: 1E-5
     *     optimizer: {@link GradientDescent Vanila Gradient Descent}
     * </pre>
     */
    public NeuralNetwork() {
        this.learningRate = 0.01;
        this.epochs = 10;
        this.batchSize = 1;
        this.threshold = 1e-5;
        layers = new ArrayList<>();

        optim = new GradientDescent(learningRate); // Set the optimizer as a standard gradient descent optimizer

        buildDetails();
    }


    /**
     * Constructs a neural network with the specified hyper-parameters.
     * Uses a default batchSize of 1 and the {@link GradientDescent Vanila Gradient Descent} optimizer.
     *
     * @param learningRate Learning to be used during optimization.
     * @param epochs Number of epochs to train the network for.
     */
    public NeuralNetwork(double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = 1;
        layers = new ArrayList<>();
        this.threshold = 1e-5;
        optim = new GradientDescent(learningRate); // Set the optimizer as a standard gradient descent optimizer

        buildDetails();
    }


    /**
     * Constructs a neural network with the specified hyper-parameters. Uses the
     * {@link GradientDescent Vanila Gradient Descent} optimizer as the default optimizer.
     *
     * @param learningRate Learning to be used during optimization.
     * @param epochs Number of epochs to train the network for.
     * @param batchSize The batch size to use during training.
     */
    public NeuralNetwork(double learningRate, int epochs, int batchSize) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        layers = new ArrayList<>();
        this.threshold = 1e-5;
        optim = new GradientDescent(learningRate); // Set the optimizer as a standard gradient descent optimizer

        buildDetails();
    }


    /**
     * Constructs a neural network with the specified hyper-parameters.
     * Uses the {@link GradientDescent Vanila Gradient Descent} optimizer as the default optimizer.
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
        layers = new ArrayList<>();

        optim = new GradientDescent(learningRate); // Set the optimizer as a standard gradient descent optimizer

        buildDetails();
    }


    /**
     * Creates a neural network with the specified hyper-parameters and optimizer. Note that when an optimizer
     * is specified, the learning rate will be specified upon creation of the optimizer and is not needed as a
     * parameter. A default batchSize of 1 will be used.
     *
     * @param optim The optimizer to use during training.
     * @param epochs The number of epochs to train the neural network for.
     */
    public NeuralNetwork(Optimizer optim, int epochs) {
        this.learningRate = optim.getLearningRate();
        this.epochs = epochs;
        this.batchSize = 1;
        this.threshold = 1e-5;
        layers = new ArrayList<>();
        this.optim = optim;
    }


    /**
     * Creates a neural network with the specified hyper-parameters and optimizer. Note that when an optimizer
     * is specified, the learning rate will be specified upon creation of the optimizer and is not needed as a
     * parameter.
     *
     * @param optim The optimizer to use during training.
     * @param epochs The number of epochs to train the neural network for.
     * @param batchSize The batch size to use during training.
     */
    public NeuralNetwork(Optimizer optim, int epochs, int batchSize) {
        this.learningRate = optim.getLearningRate();
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = 1e-5;
        layers = new ArrayList<>();
        this.optim = optim;
    }


    /**
     * Creates a neural network with the specified hyper-parameters and optimizer. Note that when an optimizer
     * is specified, the learning rate will be specified upon creation of the optimizer and is not needed as a
     * parameter.
     *
     * @param optim The optimizer to use during training.
     * @param epochs The number of epochs to train the neural network for.
     * @param batchSize The batch size to use during training.
     * @param threshold The threshold for the loss to stop early. If the loss drops below this threshold before the
     *                  specified number of epochs has been reached, the training will stop early.
     */
    public NeuralNetwork(Optimizer optim, int epochs, int batchSize, double threshold) {
        this.learningRate = optim.getLearningRate();
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.threshold = threshold;
        layers = new ArrayList<>();
        this.optim = optim;
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
        if(optim instanceof Momentum) {
            initMomentum(); // Then initialize momentum matrices.
        }

        dxUpdates = new Matrix[trainableLayers]; // Weight updates
        dxBiasUpdates = new Matrix[trainableLayers]; // bias updates
        resetDx(); // Initialize the gradient update matrices.

        Matrix feature = new Matrix(features);
        Matrix target = new Matrix(targets);
        Matrix input;
        Matrix output;
        Matrix predictions;

        for(int i=0; i<epochs; i++) {
            for(int j=0; j<feature.numRows(); j++) {
                for(int k=0; k<batchSize && (j+k)<feature.numRows(); k++) {
                    input = feature.getRowAsVector(j+k).T();
                    output = feedForward(input); // Apply the forward pass on the network.
                    back(target.getRowAsVector(j+k).T(), output, input); // Apply the backward pass of the network.
                }

                applyUpdates(); // Apply updates computed during the backward pass to the weights.
            }

            predictions = new Matrix(this.predict(features));
            lossHist.add(LossFunctions.mse.compute(predictions, target).get(0, 0).re);

            if(lossHist.get(lossHist.size()-1) < threshold) {
                break; // Then stop training since the loss has dropped below the stopping threshold.
            }

            if(optim.schedule!=null) { // Apply learning rate scheduler if applicable
                optim.schedule.step(optim);
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
            if(isFit) { // Then ensure that dropout is not applied
                if(!(layer instanceof Dropout)) { // Layer is NOT dropout so apply it.
                    currentInput = layer.forward(currentInput);
                }

            } else { // Then apply all layers on untrained model.
                currentInput = layer.forward(currentInput);
            }
        }

        return currentInput;
    }


    /**
     * Computes the backward pass of the neural network and updates weights.
     * The backwards pass computes the gradients of each layer. These gradients are used in an attempt to decrease the loss of the forward pass.
     * This is done by back-propagation and gradient descent or another optimizer.
     *
     * @param target Target for the neural network. i.e. the expected or desired output for the given input sample.
     * @param output Output of the neural network. i.e. the result of the feed forward operation.
     */
    protected void back(Matrix target, Matrix output, Matrix input) {
        Matrix error = target.sub(output); // initial error.
        Matrix[] updates;
        Matrix previousValues;
        int param_i = trainableLayers-1; // Keeps track of parameter updates index.

        for(int i=layers.size()-1; i>=1; i--) {
            if(!(layers.get(i) instanceof Dropout)) {
                // Compute the backward pass for the layer.
                updates = layers.get(i).back(layers.get(i-1).getValues(), error);

                // Update gradients
                dxUpdates[param_i] = dxUpdates[param_i].sub(updates[0]);
                dxBiasUpdates[param_i] = dxBiasUpdates[param_i].sub(updates[1]);

                error = layers.get(i).getWeights().T().mult(error);
                param_i--;
            }
        }

        if(!(layers.get(0) instanceof Dropout)) {
            updates = layers.get(0).back(input, error);

            // Update gradients
            dxUpdates[0] = dxUpdates[0].sub(updates[0]);
            dxBiasUpdates[0] = dxBiasUpdates[0].sub(updates[1]);
        }
    }


    /**
     * Applies weight updates to each layer by applying the optimizer to the weights.
     */
    private void applyUpdates() {
        int param_i = 0;

        if(optim instanceof GradientDescent) {
            for(int i=0; i<layers.size(); i++) { // Update the weights for each layer.
                // Apply the optimizer update rule to the weights and bias terms.
                if(!(layers.get(i) instanceof Dropout)) {
                    layers.get(i).setWeights(optim.step(layers.get(i).getWeights(), dxUpdates[param_i].scalDiv(batchSize)));
                    layers.get(i).setBias(optim.step(layers.get(i).getBias(), dxBiasUpdates[param_i].scalDiv(batchSize)));
                    param_i++;
                }
            }

        } else if(optim instanceof Momentum) {
            int vi = 0;

            Matrix[] wv; // Holds new weight and momentum matrices.
            Matrix[] bv; // Holds new bias and momentum matrices.

            for(int i=0; i<layers.size(); i++) { // Update the weights for each layer.
                // Apply the optimizer update rule to the weights and bias terms.

                if(!(layers.get(i) instanceof Dropout)) {
                    wv = optim.step(layers.get(i).getWeights(), dxUpdates[param_i].scalDiv(batchSize), V[vi]);
                    bv = optim.step(layers.get(i).getBias(), dxBiasUpdates[param_i].scalDiv(batchSize), V[vi+1]);

                    // Apply updates to weight, bias, and momentum matrices.
                    layers.get(i).setWeights(wv[0]);
                    layers.get(i).setBias(bv[0]);
                    V[vi] = wv[1];
                    V[vi+1] = bv[1];

                    vi+=2;
                    param_i++;
                }
            }
        } else {
            throw new IllegalStateException("Unknown optimizer: " + optim.getClass());
        }

        resetDx(); // Reset dx's for next epoch.
//        initMomentum(); // Reset momentum matrices for next epoch.
    }


    // Reset all dx's to zero.
    private void resetDx() {
        int param_i = 0;

        for(int i=0; i<layers.size(); i++) { // Initialize all weight updates to the zero matrix of appropriate size.
            if(!(layers.get(i) instanceof Dropout)) { // Ensure layer is not dropout.
                dxUpdates[param_i] = new Matrix(layers.get(i).getWeights().shape());
                dxBiasUpdates[param_i] = new Vector(layers.get(i).getOutDim());
                param_i++;
            }
        }
    }


    // initialize momentum matrices if momentum optimizer is being used.
    private void initMomentum() {
        if(!optim.name.equals(Momentum.OPTIM_NAME)) {
            throw new IllegalStateException("Can not initialize momentum vectors for optimizer " + this.optim.getClass());
        }

        V = new Matrix[trainableLayers*2]; // Create a V for each weight and bias matrix

        for(int i=0; i<V.length; i+=2) {
            V[i] = new Matrix(layers.get(i/2).getWeights().shape());
            V[i+1] = new Matrix(layers.get(i/2).getBias().shape());
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

        if(!(layer instanceof Dropout)) { // Add to the count of trainable layers.
            trainableLayers++;
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

        blockList = new Block[2 + trainableLayers];

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
            if(!(layer instanceof Dropout)) {
                blockList[count] = new Block(ModelTags.LAYER.toString(), layer.getDetails());
                count++;
            }
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
        details.append("Optimizer: " + this.optim.name + "\n");

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
