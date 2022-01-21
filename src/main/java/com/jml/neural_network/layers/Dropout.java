package com.jml.neural_network.layers;

import com.jml.core.Stats;
import com.jml.neural_network.activations.ActivationFunction;
import linalg.Matrix;
import linalg.Vector;


/**
 * A dropout layer. This layer has a probability of dropping (zeroing out) each element during the forward pass. This is
 * only done during training. Dropout layers will not be used when making predictions with the final model. <br><br>
 *
 * Dropout is an effective form of regularization. In addition, the outputs of this layer are scaled by 1/(1-p) where p is the
 * probability of dropping an element of the layer.
 */
public class Dropout implements Layer {

    public final String LAYER_TYPE = "Dropout";
    // TODO: Change mask visibility to private
    public Matrix mask; // Dropout mask
    private final double scale;

    /**
     * Probability of element being zeroed.
     */
    final double p;
    int inDim = -1; // Input size for the layer. The output size will be the same.

    // TODO: is this needed? We will just pass through the values zeroing some
    Matrix values; // Holds node values for layer.

    /**
     * Constructs a dropout layer for a neural network.<br>
     * <b><u>Note</b></u>: this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use {@link #Dropout(int, double)} to specify the input dimension.
     *
     * @param p The probability of an element being zeroed.
     */
    public Dropout(double p) {
        this.p = p;
        scale = 1/(1-p);
    }


    /**
     * Constructs a dropout layer for a neural network.
     *
     * @param inDim Input dimension of the layer.
     * @param p The probability of an element being zeroed.
     */
    public Dropout(int inDim, double p) {
        this.inDim = inDim;
        this.p = p;
        this.values = new Vector(this.inDim);
        this.mask = new Vector(this.inDim);
        scale = 1/(1-p);
    }


    /**
     * Feeds the inputs through the layer. Each input will be scaled by <code>1/(1-p)</code>
     * and have a probability, <code>p</code>, of being zeroed or "dropped."
     *
     * @param inputs Input values for the layer
     * @return The result of the layer applied to the values.
     */
    @Override
    public Matrix forward(Matrix inputs) {
        if(inputs.numRows()!=inDim && inputs.numCols()!=1) {
            throw new IllegalArgumentException("Invalid input shape of " + inputs.shape() + ". " +
                    "Expecting input shape of " + inDim + "x" + 1);
        }

        initMask(); // Initialize the mask
        values = inputs.copy().elemMult(mask).scalMult(scale); // Apply dropout mask.

        return values;
    }


    // Computes backward pass of layer.
    @Override
    public Matrix[] back(Matrix previousVals, Matrix error) {
        return new Matrix[]{previousVals.elemMult(mask).scalMult(scale)};
    }


    /**
     * {@inheritDoc}
     * @return The input dimension for this layer.
     */
    @Override
    public int getInDim() {
        return this.inDim;
    }


    /**
     * {@inheritDoc}
     * @return The output dimension for this layer.
     */
    @Override
    public int getOutDim() {
        return this.inDim;
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
        this.values = new Vector(this.inDim);
        this.mask = new Vector(this.inDim);
    }


    /**
     * Does nothing. No weights to get.
     * @return null
     */
    @Override
    public Matrix getWeights() {
        return null;
    }


    /**
     * Does nothing. No weights to set.
     * @param w New weights
     */
    @Override
    public void setWeights(Matrix w) {}


    /**
     * Does nothing. No bias to set.
     * @param b New bias
     */
    @Override
    public void setBias(Matrix b) {}


    /**
     * {@inheritDoc}
     * @return The values of this layer
     */
    @Override
    public Matrix getValues() {
        return this.values;
    }


    /**
     * Does nothing. No bias to get.
     * @return null
     */
    @Override
    public Matrix getBias() {
        return null;
    }


    // Initialize mask.
    private void initMask() {
        boolean drop;

        for(int i=0; i<mask.numRows(); i++) {
            drop = Stats.genRandBoolean(this.p);
            if(!drop) {
                mask.set(1, i, 0);
            }
        }
    }


    /**
     * Gets the details of this layer as a String.
     *
     * @return The details of this layer as a String.
     */
    @Override
    public String inspect() {
        return "Type: " + this.LAYER_TYPE + ",\tInput size: " + this.inDim + ",\tOutput size: " + this.inDim + ", \tTrainable Parameters: " + 0;
    }


    @Override
    // TODO: I think that inspect() and getDetails() names should switch.
    public String getDetails() {
        // TODO: Auto-generated method-stub
        return null;
    }

    @Override
    public ActivationFunction getActivation() {
        return null;
    }
}
