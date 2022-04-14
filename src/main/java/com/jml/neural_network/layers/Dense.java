package com.jml.neural_network.layers;

import com.jml.core.Block;
import com.jml.neural_network.ModelTags;
import com.jml.neural_network.activations.ActivationFunction;
import com.jml.neural_network.activations.Softmax;
import com.jml.neural_network.layers.initilizers.GlorotNormal;
import com.jml.neural_network.layers.initilizers.Initializer;
import com.jml.neural_network.layers.initilizers.Zeros;
import com.jml.util.ArrayUtils;
import linalg.Matrix;


/**
 * A fully connected layer with an {@link com.jml.neural_network.activations.ActivationFunction activation function}.
 * For an activation function g(x), this layer applies the transform f(x) = g(Wx+b) for an input vector x.
 */
public class Dense extends Linear {
    public final String LAYER_TYPE = "Dense";

    protected ActivationFunction activation;

    /**
     * Creates a Linear layer with specified input and output dimensions.
     *
     * @param inDim  Input dimension for this layer. (i.e. the number of nodes in the previous layer)
     * @param outDim Output dimension for this layer. (i.e. the number of nodes in this layer.)
     * @param activation Activation function for this layer.
     */
    public Dense(int inDim, int outDim, ActivationFunction activation) {
        super(inDim, outDim);
        this.activation = activation;
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.
     *
     * @param inDim             Input dimension for this layer. (i.e. the number of nodes in the previous layer)
     * @param outDim            Output dimension for this layer. (i.e. the number of nodes in this layer.)
     * @param activation Activation function for this layer.
     * @param weightInitializer Initializer for the weights of this layer.
     */
    public Dense(int inDim, int outDim, ActivationFunction activation, Initializer weightInitializer) {
        super(inDim, outDim, weightInitializer);
        this.activation = activation;
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.
     *
     * @param inDim             Input dimension for this layer. (i.e. the number of nodes in the previous layer)
     * @param outDim            Output dimension for this layer. (i.e. the number of nodes in this layer.)
     * @param activation Activation function for this layer.
     * @param weightInitializer Initializer for the weights of this layer.
     * @param biasInitializer   Initializer for the bias terms of this layer.
     */
    public Dense(int inDim, int outDim, ActivationFunction activation, Initializer weightInitializer, Initializer biasInitializer) {
        super(inDim, outDim, weightInitializer, biasInitializer);
        this.activation = activation;
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.<br>
     * <b>NOTE:</b> this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use todo...
     *
     * @param outDim Output dimension for this layer (i.e. the number of nodes in this layer).
     * @param activation Activation function for this layer.
     */
    public Dense(int outDim, ActivationFunction activation) {
        super(outDim);
        this.activation = activation;
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.<br>
     * <b>NOTE:</b> this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use todo...
     *
     * @param outDim            Output dimension for this layer (i.e. the number of nodes in this layer).
     * @param activation Activation function for this layer.
     * @param weightInitializer Initializer for the weights of this layer.
     */
    public Dense(int outDim, ActivationFunction activation, Initializer weightInitializer) {
        super(outDim, weightInitializer);
        this.activation = activation;
    }


    /**
     * Creates a Linear layer with specified input and output dimensions.<br>
     * <b>NOTE:</b> this constructor infers the input dimension from the
     * previous layer in the network. Thus, it cannot be used as the first layer of the neural network. For the first
     * layer use todo...
     *
     * @param outDim            Output dimension for this layer (i.e. the number of nodes in this layer).
     * @param activation Activation function for this layer.
     * @param weightInitializer Initializer for the weights of this layer.
     * @param biasInitializer   Initializer for the bias terms of this layer.
     */
    public Dense(int outDim, ActivationFunction activation, Initializer weightInitializer, Initializer biasInitializer) {
        super(outDim, weightInitializer, biasInitializer);
        this.activation = activation;
    }


    /**
     * Computes the forward pass of this layer.
     * @param input Input to this Layer. Must be a column vector.
     * @return Result of the forward pass of this layer as a Matrix.
     */
    @Override
    public Matrix forward(Matrix input) {
        return activation.forward(super.forward(input)); // Apply activation to linear transform.
    }


    /**
     * Computes the backward pass for this layer.
     *
     * @param upstreamGrad Upstream gradient of the network.
     * @return Result of the backwards pass of the layer as a Matrix. If this layer has weights, this matrix
     * will have the same shape as the weight matrix for the layer.
     */
    public Matrix back(Matrix upstreamGrad) {

        if(this.activation instanceof Softmax) {
            // TODO:
        } else {
            Matrix commonGrad = upstreamGrad.elemMult(activation.back(this.forwardOut));
            this.wGrad = commonGrad.mult(this.forwardIn.T());
            this.bGrad = upstreamGrad.sumCols();
            this.backwardOut = weights.T().mult(commonGrad);
        }

        System.out.println("wGrad:\n" + wGrad);
        System.out.println("bGrad:\n" + bGrad + "\n\n");

        return backwardOut;
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
        StringBuilder inspection = new StringBuilder("Type: " + LAYER_TYPE + ",\tInput size: "
                + inDim + ",\tOutput size: " + outDim + ", \tTrainable Parameters: " + (inDim*outDim + outDim) +
                ",\tActivationFunction: " + activation.getName());
        return inspection.toString();
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
        Block activationBlock = new Block(ModelTags.ACTIVATION.toString(), this.activation.getName());
        Block dimBlock = new Block(ModelTags.DIMENSIONS.toString(), this.inDim + ", " + this.outDim);
        Block weightBlock = new Block(ModelTags.WEIGHTS.toString(), ArrayUtils.asString(this.weights.getValuesAsDouble()));
        Block biasBlock = new Block(ModelTags.BIAS.toString(), ArrayUtils.asString(this.bias.getValuesAsDouble()));

        // Combine all blocks into a single string.
        details.append(Block.buildFileContent(layerBlock, activationBlock, dimBlock, weightBlock, biasBlock));

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