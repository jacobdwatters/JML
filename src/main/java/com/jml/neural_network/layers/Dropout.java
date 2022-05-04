package com.jml.neural_network.layers;

import com.jml.core.Block;
import com.jml.core.Stats;
import com.jml.neural_network.ModelTags;
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
    protected Matrix mask; // Dropout mask
    private final double scale;

    protected Matrix forwardIn;
    protected Matrix forwardOut;
    protected Matrix backwardOut;

    /**
     * Probability of element being zeroed.
     */
    final double p;
    int inDim = -1; // Input size for the layer. The output size will be the same.

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
        this.forwardIn = new Vector(this.inDim);
        this.mask = new Vector(this.inDim);
        scale = 1/(1-p);
    }


    /**
     * Feeds the inputs through the layer. Each input will be scaled by <code>1/(1-p)</code>
     * and have a probability, <code>p</code>, of being zeroed or "dropped."
     *
     * @param forwardIn Input values for the layer
     * @return The result of the layer applied to the values.
     */
    @Override
    public Matrix forward(Matrix forwardIn) {
        if(forwardIn.numRows()!=inDim && forwardIn.numCols()!=1) {
            throw new IllegalArgumentException("Invalid input shape of " + forwardIn.shape() + ". " +
                    "Expecting input shape of " + inDim + "x" + 1);
        }

        initMask(forwardIn.numCols()); // Initialize the mask
        this.forwardOut = forwardIn.elemMult(mask).scalMult(scale); // Apply dropout mask.

        return forwardOut;
    }


    // Computes backward pass of layer.
    @Override
    public Matrix back(Matrix upstreamGrad) {
        this.backwardOut = upstreamGrad;
        return backwardOut;
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
        this.forwardIn = new Vector(this.inDim);
        this.mask = new Vector(this.inDim);
    }


    // Initialize dropout mask.
    private void initMask(int size) {
        boolean drop;

        this.mask = new Vector(this.inDim);

        for(int i=0; i<this.mask.numRows(); i++) {
            drop = Stats.genRandBoolean(this.p);
            if(!drop) {
                this.mask.set(1, i, 0);
            }
        }

        this.mask = this.mask.extend(size);
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


    /**
     * Constructs a string containing this all details of the model pertinent for saving the model to a file.
     *
     * @return A string containing all information, including trainable parameters, needed to recreate the layer.
     */
    @Override
    public String getDetails() {
        StringBuilder details = new StringBuilder();

        // Create all the blocks for this layer.
        Block layerBlock = new Block(ModelTags.TYPE.toString(), this.LAYER_TYPE);
        Block dimBlock = new Block(ModelTags.DIMENSIONS.toString(), this.inDim + ", " + this.inDim);
        Block pBlock = new Block(ModelTags.PROBABILITY.toString(), Double.toString(this.p));

        // Combine all blocks into a single string.
        details.append(Block.buildFileContent(layerBlock, dimBlock, pBlock));

        return details.toString();
    }
}
