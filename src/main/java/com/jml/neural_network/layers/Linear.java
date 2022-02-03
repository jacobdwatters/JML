package com.jml.neural_network.layers;

import com.jml.core.Block;
import com.jml.neural_network.ModelTags;
import com.jml.neural_network.layers.initilizers.*;
import com.jml.util.ArrayUtils;
import linalg.Matrix;


/**
 * Simple fully-connected linear layer. Applies the linear transform f(x)=Wx+b for an input vector x.
 * This is equivalent to a {@link Dense} layer with a {@link com.jml.neural_network.activations.Linear linear} activation.
 * <br><br>
 *
 * - The default weightInitializer is the {@link GlorotNormal} initializer. <br>
 * - The default biasInitializer is the {@link Zeros} initializer.
 */
public class Linear implements TrainableLayer {
    public final String LAYER_TYPE = "Linear";

    protected int inDim, outDim;

    protected Matrix weights;
    protected Matrix bias;

    protected Initializer weightInitializer;
    protected Initializer biasInitializer;

    private Matrix forwardIn; // Inputs to this layer. i.e. values of previous layers nodes.
    private Matrix forwardOut; // Output of this layer. i.e. values of this layers' nodes.
    private Matrix backwardOut; // New Upstream gradient to use for backpropagation computations in previous layer.

    private Matrix wGrad; // Gradient of model with respect to the weights of this layer.
    private Matrix bGrad; // Gradients of model with respect to the bias terms of this layer.


    /**
     * Creates a Linear layer with specified input and output dimensions.
     * @param inDim Input dimension for this layer. (i.e. the number of nodes in the previous layer)
     * @param outDim Output dimension for this layer. (i.e. the number of nodes in this layer.)
     */
    public Linear(int inDim, int outDim) {
        this.inDim = inDim;
        this.outDim = outDim;

        this.weightInitializer = new GlorotNormal();
        this.biasInitializer = new Zeros();

        initLayer(); // Initialize bias and weights
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.
     * @param inDim Input dimension for this layer. (i.e. the number of nodes in the previous layer)
     * @param outDim Output dimension for this layer. (i.e. the number of nodes in this layer.)
     * @param weightInitializer Initializer for the weights of this layer.
     */
    public Linear(int inDim, int outDim, Initializer weightInitializer) {
        this.inDim = inDim;
        this.outDim = outDim;

        this.weightInitializer = weightInitializer;
        this.biasInitializer = new Zeros();

        initLayer(); // Initialize bias and weights
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.
     * @param inDim Input dimension for this layer. (i.e. the number of nodes in the previous layer)
     * @param outDim Output dimension for this layer. (i.e. the number of nodes in this layer.)
     * @param weightInitializer Initializer for the weights of this layer.
     * @param biasInitializer Initializer for the bias terms of this layer.
     */
    public Linear(int inDim, int outDim, Initializer weightInitializer, Initializer biasInitializer) {
        this.inDim = inDim;
        this.outDim = outDim;

        this.weightInitializer = weightInitializer;
        this.biasInitializer = biasInitializer;

        initLayer(); // Initialize bias and weights
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.<br>
     * <b>NOTE:</b> this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Linear(int, int)}.
     * @param outDim Output dimension for this layer (i.e. the number of nodes in this layer).
     */
    public Linear(int outDim) {
        this.inDim = -1;
        this.outDim = outDim;

        this.weightInitializer = new GlorotNormal();
        this.biasInitializer = new Zeros();
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.<br>
     * <b>NOTE:</b> this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Linear(int, int, Initializer)}.
     * @param outDim Output dimension for this layer (i.e. the number of nodes in this layer).
     * @param weightInitializer Initializer for the weights of this layer.
     */
    public Linear(int outDim, Initializer weightInitializer) {
        this.inDim = -1;
        this.outDim = outDim;

        this.weightInitializer = weightInitializer;
        this.biasInitializer = new Zeros();
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.<br>
     * <b>NOTE:</b> this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Linear(int, int, Initializer, Initializer)}.
     * @param outDim Output dimension for this layer (i.e. the number of nodes in this layer).
     * @param weightInitializer Initializer for the weights of this layer.
     * @param biasInitializer Initializer for the bias terms of this layer.
     */
    public Linear(int outDim, Initializer weightInitializer, Initializer biasInitializer) {
        this.inDim = -1;
        this.outDim = outDim;

        this.weightInitializer = weightInitializer;
        this.biasInitializer = biasInitializer;
    }


    /**
     * Computes forward pass for layer.
     *
     * @param input Input to this Layer. Must be a column vector.
     * @return Result of the forward pass of a layer as a matrix.
     */
    @Override
    public Matrix forward(Matrix input) {
        forwardIn = input;
        forwardOut = weights.mult(input);

        return forwardOut;
    }


    /**
     * Computes backward pass for layer.
     *
     * @param upstreamGrad Upstream gradient of the network.
     * @return Result of the backwards pass of the layer as a Matrix. If this layer has weights, this matrix
     * will have the same shape as the weight matrix for the layer.
     */
    @Override
    public Matrix back(Matrix upstreamGrad) {
        this.wGrad = upstreamGrad.T().mult(this.forwardIn.T());
        this.bGrad = upstreamGrad.T();

        this.backwardOut = upstreamGrad.mult(weights); // Gradient of model with respect to inputs of this layer.

        return backwardOut;
    }


    /**
     * Resists the accumulation of gradients for this layer.
     */
    @Override
    public void resetGradients() {
        this.wGrad = new Matrix(this.outDim, this.inDim); // Reset weight updates to 0.
        this.bGrad = new Matrix(this.outDim, 1); // Reset bias updates to 0.
    }


    /**
     * Gets the input dimension of this layer.
     *
     * @return The input dimension of this layer.
     */
    @Override
    public int getInDim() {
        return this.inDim;
    }


    /**
     * Gets the output dimension of this layer.
     *
     * @return The output dimension of this layer.
     */
    @Override
    public int getOutDim() {
        return this.outDim;
    }


    /**
     * Updates this layers input dimension. This is useful for creating a layer with an unknown input dimension and
     * inferring it from the previous layer in the network.
     */
    @Override
    public void updateInDim(int newInDim) {
        this.inDim = newInDim;
        initLayer(); // Initialize weights and bias
    }


    /**
     * Gets the trainable parameters for this layer as an array of matrices.
     *
     * @return The trainable parameters for this layer as an array of matrices {weights, bias}.
     */
    @Override
    public Matrix[] getParams() {
        return new Matrix[]{weights, bias};
    }


    /**
     * Gets the update matrices for parameters of this layer.
     * @return The parameter update matrices.
     */
    @Override
    public Matrix[] getUpdates() {
        return new Matrix[]{wGrad, bGrad};
    }


    /**
     * Sets the parameters for this layer.
     *
     * @param params Parameter Matrices for this layer.
     */
    @Override
    public void setParams(Matrix... params) {
        if(params.length!=2) {
            throw new IllegalArgumentException("Expecting 2 parameter matrices for Linear layer of form {Weights, bias} but " +
                    "got " + params.length + ".");
        }
        if(!params[0].sameShape(weights)) {
            throw new IllegalArgumentException("First parameter matrix does not have same shape as weight matrix. Expecting "
            + weights.shape() + " but got " + params[0].shape() + ".");
        }
        if(!params[1].sameShape(bias)) {
            throw new IllegalArgumentException("Second parameter matrix does not have same shape as bias matrix. Expecting "
                    + bias.shape() + " but got " + params[1].shape() + ".");
        }

        this.weights = params[0];
        this.bias = params[1];
    }


    /**
     * Gets and formats details of this layer in a human-readable String.
     * @return The details of this layer in a human-reusable String.
     */
    @Override
    public String inspect() {
        StringBuilder details = new StringBuilder("Type: " + LAYER_TYPE + ",\tInput size: "
                + inDim + ",\tOutput size: " + outDim + ", \tTrainable Parameters: " + (inDim*outDim + outDim));
        return details.toString();
    }


    /**
     * Initializes the weights for this layer.
     */
    private void initLayer() {
        this.weights = weightInitializer.init(this.outDim, this.inDim); // Initialize weight values.
        this.bias = biasInitializer.init(this.outDim, 1); // Initialize bias values.
        resetGradients(); // Set weight and bias updates to 0.
    }


    /**
     * Constructs a string containing this all details of the model pertinent for saving the model to a file.
     *
     * @return A string containing all information, including trainable parameters, needed to recreate the layer.
     */
    @Override
    public String getDetails() {
        // TODO: add optimizer block
        StringBuilder details = new StringBuilder();

        // Create all the blocks for this layer.
        Block layerBlock = new Block(ModelTags.TYPE.toString(), this.LAYER_TYPE);
        Block dimBlock = new Block(ModelTags.DIMENSIONS.toString(), this.inDim + ", " + this.outDim);
        Block weightBlock = new Block(ModelTags.WEIGHTS.toString(), ArrayUtils.asString(this.weights.getValuesAsDouble()));
        Block biasBlock = new Block(ModelTags.BIAS.toString(), ArrayUtils.asString(this.bias.getValuesAsDouble()));

        // Combine all blocks into a single string.
        details.append(Block.buildFileContent(layerBlock, dimBlock, weightBlock, biasBlock));

        return details.toString();
    }


    /**
     * Gets the name of the layer type as a string.
     * @return The type of this layer as a String.
     */
    public String toString() {
        return LAYER_TYPE;
    }
}
