package com.jml.neural_network;

import com.jml.core.Block;
import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.neural_network.layers.BaseLayer;
import com.jml.neural_network.layers.Dropout;
import com.jml.neural_network.layers.TrainableLayer;
import com.jml.optimizers.Adam;
import com.jml.optimizers.GradientDescent;
import com.jml.optimizers.Momentum;
import com.jml.optimizers.Optimizer;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.List;


/**
 * A class that supports the creation and training of neural networks. A neural network is a supervised learning model that is structured in layers.<br>
 * Neural networks are created sequentially one layer at a time by using the {@link #add(BaseLayer)} method. Activation functions can be specified for applicable layers.<br>
 *
 * <pre>
 * Built-in Layers:
 *      {@link com.jml.neural_network.layers.Dense Dense}
 *      {@link com.jml.neural_network.layers.Linear Linear}
 *      {@link Dropout Dropout}</pre>
 *
 * <pre>
 * Built-in Activation Functions:
 *      {@link com.jml.neural_network.activations.Linear Linear}
 *      {@link com.jml.neural_network.activations.Sigmoid Sigmoid}
 *      {@link com.jml.neural_network.activations.Relu ReLU}
 *      {@link com.jml.neural_network.activations.Tanh Tanh}
 *      {@link com.jml.neural_network.activations.Softmax Softmax}</pre>
 *
 * Custom layers or activation functions can be created using the ActivationFunction, BaseLayer, and TrainableLayer interfaces.
 */
public class NeuralNetwork extends Model<double[][], double[][]> {
    protected final String MODEL_TYPE = ModelTypes.NEURAL_NETWORK.toString();
    protected final List<BaseLayer> layers;
    protected double learningRate;
    protected double threshold;
    protected int epochs;
    protected int batchSize;

    private int trainableLayers=0;

    protected boolean isFit = false;
    final List<Double> lossHist = new ArrayList<>();


    // TODO: Should these be moved to the layer? Probably yes!
    private Matrix[] V; // Momentum update matrices. Only used for the Momentum and Adam optimizers.
    private Matrix[] M; // Adam moment update matrices.

    protected final Optimizer optim; // Optimizer to use during backpropagation.

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
        if(this.layers.size()==0) {
            throw new IllegalStateException("Neural Network has no layers. Neural Networks with no layers can not be trained.");
        }

        if(optim instanceof Momentum) {
            initMomentum(); // Then initialize momentum matrices.
        } else if(optim instanceof Adam) {
            initAdam(); // Then initialize Adam matrices.
        }

        double[][][] shuffle;
        Matrix feature;
        Matrix target = new Matrix(targets);
        Matrix input;
        Matrix output;
        Matrix predictions;

        predictions = new Matrix(this.predict(features));
        lossHist.add(LossFunctions.mse.compute(predictions, target).get(0, 0).re); // Beginning loss.

        for(int i=0; i<epochs; i++) {
            shuffle = ArrayUtils.shuffle(features, targets); // Shuffle samples for this epoch.
            feature = new Matrix(shuffle[0]);
            target = new Matrix(shuffle[1]);

            for(int j=0; j<feature.numRows(); j+=batchSize) { // Iterate over all samples
                for(int k=0; k<batchSize && (j+k)<feature.numRows(); k++) { // Iterate over the batch.
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
        for(BaseLayer layer : layers) { // Feeds the input through all layers.
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
        /* TODO: Initial upstreamGrad is currently the derivative of MSE but should be the derivative of any loss function.
                Should allow the use of a specified loss function and replace this initial upstreamGrad with the derivative
                 of the loss function.*/
        Matrix upstreamGrad = output.sub(target).T(); // initial upstream gradient.

        int param_i = trainableLayers-1; // Keeps track of parameter updates index.

        for(int i=layers.size()-1; i>=1; i--) {

            if(layers.get(i) instanceof TrainableLayer) { // Apply backdrop to trainable layer.
                // Compute the backward pass for the layer.
                upstreamGrad = layers.get(i).back(upstreamGrad);
                param_i--;
            }
        }

        if(!(layers.get(0) instanceof Dropout)) {
            layers.get(0).back(upstreamGrad);
        }
    }


    /**
     * Applies weight updates to each layer by applying the optimizer to the weights.
     */
    private void applyUpdates() {
        Matrix[] params; // Holds trainable parameters for a layer.
        Matrix[] updates; // Holds update matrices for the parameters of this layer.

        if(optim instanceof GradientDescent) {
            Matrix newW, newB;

            for(BaseLayer layer : layers) { // Update the weights for each layer.
                // Apply the optimizer update rule to the weights and bias terms.
                if(layer instanceof TrainableLayer) {
                    params = layer.getParams();
                    updates = layer.getUpdates();

                    newW = optim.step(params[0], updates[0].scalDiv(batchSize))[0];
                    newB = optim.step(params[1], updates[1].scalDiv(batchSize))[0];

                    layer.setParams(newW, newB);
                    layer.resetGradients();
                }
            }

        } else if(optim instanceof Momentum) {
            int vi = 0;

            Matrix[] wv; // Holds new weight and momentum matrices.
            Matrix[] bv; // Holds new bias and momentum matrices.

            for(BaseLayer layer : layers) { // Update the weights for each layer.
                // Apply the optimizer update rule to the weights and bias terms.

                if(layer instanceof TrainableLayer) {
                    params = layer.getParams();
                    updates = layer.getUpdates();

                    wv = optim.step(params[0], updates[0].scalDiv(batchSize), V[vi]);
                    bv = optim.step(params[1], updates[1].scalDiv(batchSize), V[vi+1]);

                    // Apply updates to weight, bias, and momentum matrices.
                    layer.setParams(wv[0], bv[0]);
                    layer.resetGradients();

                    V[vi] = wv[1];
                    V[vi+1] = bv[1];

                    vi+=2;
                }
            }

        } else if(optim instanceof Adam) {
            int vi = 0;

            Matrix[] wvm; // Holds new weight and momentum matrices.
            Matrix[] bvm; // Holds new bias and momentum matrices.

            for(BaseLayer layer : layers) { // Update the weights for each layer.
                // Apply the optimizer update rule to the weights and bias terms.

                if(layer instanceof TrainableLayer) {
                    params = layer.getParams();
                    updates = layer.getUpdates();

                    wvm = optim.step(vi==0, params[0], updates[0].scalDiv(batchSize),
                            V[vi], M[vi]);
                    bvm = optim.step(false, params[1], updates[1].scalDiv(batchSize),
                            V[vi+1], M[vi+1]);

                    // Apply updates to weight, bias, and momentum matrices.
                    layer.setParams(wvm[0], bvm[0]);
                    layer.resetGradients();

                    V[vi] = wvm[1];
                    V[vi+1] = bvm[1];

                    M[vi] = wvm[2];
                    M[vi+1] = bvm[2];

                    vi+=2;
                }
            }

        } else {
            throw new IllegalStateException("Unknown optimizer: " + optim.getClass());
        }
    }


    // initialize momentum matrices if momentum optimizer is being used.
    private void initMomentum() {
        if(!(optim instanceof Momentum)) {
            throw new IllegalStateException("Can not initialize momentum vectors for optimizer " + this.optim.getClass());
        }

        V = new Matrix[trainableLayers*2]; // Create a V for each weight and bias matrix
        int vi = 0;

        for(int i=0; i<layers.size(); i++) {

            if(layers.get(i) instanceof TrainableLayer) { // Ensure layer is trainable.
                V[vi] = new Matrix(layers.get(i).getOutDim(), layers.get(i).getInDim()); // For weights
                V[vi+1] = new Vector(layers.get(i).getOutDim()); // For bias terms
                vi+=2;
            }
        }
    }


    // initialize moment matrices if Adam optimizer is being used.
    private void initAdam() {
        if(!(optim instanceof Adam)) {
            throw new IllegalStateException("Can not initialize Adam vectors for optimizer " + this.optim.getClass());
        }

        V = new Matrix[trainableLayers*2]; // Create a V for each weight and bias matrix
        M = new Matrix[trainableLayers*2]; // Create a M for each weight and bias matrix
        int vi = 0;

        for(int i=0; i<layers.size(); i++) {

            if(layers.get(i) instanceof TrainableLayer) { // Ensure layer is trainable.

                V[vi] = new Matrix(layers.get(i).getOutDim(), layers.get(i).getInDim()); // For weights
                V[vi+1] = new Vector(layers.get(i).getOutDim()); // For bias terms

                M[vi] = new Matrix(layers.get(i).getOutDim(), layers.get(i).getInDim()); // For weights
                M[vi+1] = new Vector(layers.get(i).getOutDim()); // For bias terms

                vi+=2;
            }
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
    public void add(BaseLayer layer) {
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

        if(layer instanceof TrainableLayer) {
            trainableLayers++; // Add to the count of trainable layers.
        }

        layers.add(layer);
        buildDetails();
    }


    public List<Double> getLossHist() {
        return lossHist;
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

        blockList = new Block[2 + trainableLayers];

        StringBuilder hyperParams = new StringBuilder();
        hyperParams.append(this.learningRate).append(", ");
        hyperParams.append(this.epochs).append(", ");
        hyperParams.append(this.batchSize).append(", ");
        hyperParams.append(this.threshold);

        // Construct the blocks for the model file.
        blockList[0] = new Block(ModelTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(ModelTags.HYPER_PARAMETERS.toString(), hyperParams.toString());
        // TODO: Add getDetails() in optimizer so that it can be saved in the model.
        //        blockList[2] = new Block(ModelTags.OPTIMIZER.toString(), optim.getDetails())

        int count = 2;
        StringBuilder layerDetails = new StringBuilder();

        for(BaseLayer layer : layers) {
            if(!(layer instanceof Dropout)) { // TODO: save dropout as well.
                blockList[count] = new Block(ModelTags.LAYER.toString(), layer.getDetails());
                count++;
            }
        }

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }


    /**
     * Builds the details of this model. 'Details' includes all information needed to recreate the model.
     */
    protected void buildDetails() {
        details = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE+ "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );

        details.append("Learning Rate: ").append(this.learningRate).append("\n");
        details.append("Batch Size: ").append(this.batchSize).append("\n");
        details.append("Optimizer: ").append(this.optim.name).append("\n");

        if(!layers.isEmpty()) {
            details.append("Layers (").append(layers.size()).append("):\n").append("------------\n");

            int layerCount = 1;
            for(BaseLayer layer : this.layers) {
                details.append("\t").append(layerCount).append("\t").append(layer.inspect()).append("\n");
                layerCount++;
            }
        }
    }


    /**
     * Forms a string of the important aspects of the model which are needed to recreate the model.<br>
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
