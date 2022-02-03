package com.jml.neural_network.layers;

import com.jml.neural_network.layers.BaseLayer;
import linalg.Matrix;


/**
 * This layer specifies functionality for a layer that has trainable parameters.
 */
public interface TrainableLayer extends BaseLayer {

    /**
     * Gets the trainable parameters for this layer as an array of matrices.
     * @return The trainable parameters for this layer as an array of matrices.
     */
    @Override
    Matrix[] getParams();


    /**
     * Sets the parameters for this layer.
     * @param params Parameter matrices for this layer.
     */
    @Override
    void setParams(Matrix... params);


    /**
     * Resists the accumulation of gradients for this layer.
     */
    @Override
    void resetGradients();


    /**
     * Gets the update matrices for parameters of this layer.
     * @return The parameter update matrices.
     */
    @Override
    Matrix[] getUpdates();
}
