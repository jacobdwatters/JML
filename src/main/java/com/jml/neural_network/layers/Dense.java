package com.jml.neural_network.layers;

import com.jml.core.Block;
import com.jml.neural_network.ModelTags;
import com.jml.neural_network.activations.ActivationFunction;
import com.jml.util.ArrayUtils;
import linalg.Matrix;
import linalg.Vector;


/**
 * A standard dense or fully connected neural network layer.
 * This layer applies a linear transform <code>y=f(Ax+b)</code> onto the inputs x where f is the
 * {@link com.jml.neural_network.activations.ActivationFunction activation function} used in the layer.
 */
public class Dense implements Layer {

    // TODO: Add ability to specify weight initializer. Allow for one to be specified for bias and weights individually

    public final String LAYER_TYPE = "Dense";
    public int inDim = -1; // Size of the input.
    public int outDim; // Size of the output.
    public ActivationFunction activation; // ActivationFunction function for the layer.
    protected int paramCount;

    Matrix values; // Node values for the layer
    Matrix weights; // Weights for the layer.
    Matrix bias; // Bias for the layer.

    private double std = 0.5;

    /**
     * Constructs a dense layer for a neural network.<br>
     * <b><u>Note</b></u>: this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Dense(int, int, ActivationFunction)} to specify the input dimension.
     *
     * @param outDim Layer output dimension.
     * @param activation ActivationFunction function for layer.
     */
    public Dense(int outDim, ActivationFunction activation) {
        if(outDim<=0) {
            throw new IllegalArgumentException("Expecting outDim to be positive but got " + outDim);
        }

        this.outDim = outDim;
        this.activation = activation;

        this.bias = new Matrix(this.outDim, 1);
    }


    /**
     * Constructs a dense layer for a neural network.
     *
     * @param inDim Layer input dimension.
     * @param outDim Layer output dimension.
     * @param activation ActivationFunction function for layer.
     */
    public Dense(int inDim, int outDim, ActivationFunction activation) {
        if(outDim<=0) {
            throw new IllegalArgumentException("Expecting outDim to be positive but got " + outDim);
        }
        if(inDim<=0) {
            throw new IllegalArgumentException("Expecting inDim to be positive but got " + inDim);
        }

        this.inDim = inDim;
        this.outDim = outDim;
        this.activation = activation;
        this.paramCount = inDim*outDim+outDim;

        this.values = new Vector(this.outDim);

        this.weights = Initializers.randn(this.outDim, this.inDim, this.std); // Initialize weights.
        this.bias = new Matrix(this.outDim, 1); // Initialize bias terms to zero.
    }


    /**
     * {@inheritDoc}
     *
     * @param inputs Input values for the layer.
     */
    @Override
    public Matrix forward(Matrix inputs) {
        values = activation.apply(weights.mult(inputs).add(bias));
        return values;
    }


    // Computes backward pass of the layer.
    /**
     * {@inheritDoc}
     * @param previousVals Values of the previous layer in the network (or input values for first layer).
     * @param error Error for this layer.
     * @return The weight and bias updates for this layer in a Matrix array.
     */
    @Override
    public Matrix[] back(Matrix previousVals, Matrix error) {

        Matrix dxUpdates = activation.slope(values)
                        .elemMult(error)
                        .mult(previousVals.T());

        Matrix dxBiasUpdates = activation.slope(bias)
                        .elemMult(error);

        return new Matrix[]{dxUpdates, dxBiasUpdates};
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public int getInDim() {
        return this.inDim;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public int getOutDim() {
        return this.outDim;
    }


    /**
     * Updates the input dimension for the layer. <br>
     * <b><u>WARNING:</u></b> This will zero any weight values the layer may currently be holding.
     *
     * @param inDim New input size for the layer.
     */
    @Override
    public void updateInDim(int inDim) {
        this.inDim = inDim;
        this.weights = Initializers.randn(this.outDim, this.inDim, std); // Initialize weights
        this.paramCount = inDim*outDim + outDim;
    }


    /**
     * Gets the weights of this layer.
     * @return Returns the weights of this layer in a 2D array.
     */
    @Override
    public Matrix getWeights() {
        return this.weights;
    }


    @Override
    public void setWeights(Matrix w) {
        this.weights = w.copy();
    }


    @Override
    public void setBias(Matrix b) {this.bias = b.copy();}


    /**
     * {@inheritDoc}
     * @return The values of this layer
     */
    @Override
    public Matrix getValues() {
        return this.values;
    }


    @Override
    public Matrix getBias() {
        return bias;
    }

    /**
     * Gets the details of this layer as a String.
     *
     * @return The details of this layer as a String.
     */
    @Override
    public String inspect() {
        StringBuilder details = new StringBuilder("Type: " + LAYER_TYPE + ",\tInput size: "
                + inDim + ",\tOutput size: " + outDim + ", \tTrainable Parameters: " + paramCount +
                ",\tActivationFunction: " + activation.getName());
        return details.toString();
    }


    /**
     * Constructs a string containing this layers activation function, input/output dimension,
     * and all parameters of the layer. Note, this will be more detailed than the getDetails() method and
     * can result in large strings.
     *
     * @return A string containing all information, including trainable parameters needed to recreate the layer.
     */
    @Override
    public String getDetails() {
        StringBuilder inspection = new StringBuilder();

        // Create all the blocks for this layer.
        Block layerBlock = new Block(ModelTags.TYPE.toString(), this.LAYER_TYPE);
        Block activationBlock = new Block(ModelTags.ACTIVATION.toString(), this.activation.getName());
        Block dimBlock = new Block(ModelTags.DIMENSIONS.toString(), this.inDim + ", " + this.outDim);
        Block weightBlock = new Block(ModelTags.WEIGHTS.toString(), ArrayUtils.asString(this.weights.getValuesAsDouble()));
        Block biasBlock = new Block(ModelTags.BIAS.toString(), ArrayUtils.asString(this.bias.getValuesAsDouble()));

        // Combine all blocks into a single string.
        inspection.append(Block.buildFileContent(layerBlock, activationBlock, dimBlock, weightBlock, biasBlock));

        return inspection.toString();
    }

    @Override
    public ActivationFunction getActivation() {
        return this.activation;
    }
}
