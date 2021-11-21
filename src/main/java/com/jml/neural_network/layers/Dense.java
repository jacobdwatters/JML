package com.jml.neural_network.layers;

import com.jml.neural_network.activations.Activation;
import linalg.Matrix;
import linalg.Vector;


/**
 * A standard dense or fully connected neural network layer.
 * This layer applies a linear transform <code>y=Ax+b</code> onto the inputs x.
 */
public class Dense implements Layer {
    public final String LAYER_TYPE = "Dense";
    public int inDim = -1; // Size of the input.
    public int outDim; // Size of the output.
    public Activation activation; // Activation function for the layer.
    protected int paramCount;

    Matrix values; // Node values for the layer
    Matrix weights; // Weights for the layer.
    Matrix bias; // Bias for the layer.


    /**
     * Constructs a dense layer for a neural network.<br>
     * <b><u>Note</b></u>: this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Dense(int, int, Activation)} to specify the input dimension.
     *
     * @param outDim Layer output dimension.
     * @param activation Activation function for layer.
     */
    public Dense(int outDim, Activation activation) {
        if(outDim<=0) {
            throw new IllegalArgumentException("Expecting outDim to be positive but got " + outDim);
        }

        this.outDim = outDim;
        this.activation = activation;

        this.bias = Matrix.random(this.outDim, 1);
    }


    /**
     * Constructs a dense layer for a neural network.
     *
     * @param inDim Layer input dimension.
     * @param outDim Layer output dimension.
     * @param activation Activation function for layer.
     */
    public Dense(int inDim, int outDim, Activation activation) {
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
        this.weights = Matrix.random(this.outDim, this.inDim, -10, 10); // Initialize weights

        this.bias = Matrix.random(this.outDim, 1,-10, 10);
    }


    /**
     * {@inheritDoc}
     *
     * @param inputs Input values for the layer.
     */
    @Override
    public Matrix forward(Matrix inputs) {
        values = activation.apply(weights.mult(inputs).add(bias)); // TODO: Add bias
        return values;
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
        this.weights = Matrix.random(this.outDim, this.inDim, -10, 10); // Initialize weights
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
    public String getDetails() {
        StringBuilder details = new StringBuilder("Type: " + LAYER_TYPE + ",\tInput size: "
                + inDim + ",\tOutput size: " + outDim + ", \tTrainable Parameters: " + paramCount);
        return details.toString();
    }
}
