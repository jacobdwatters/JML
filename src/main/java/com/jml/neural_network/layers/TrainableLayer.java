package com.jml.neural_network.layers;

import linalg.Matrix;


/**
 * This interface specifies functionality for a layer that has trainable parameters.
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


    /**
     * Checks if this layer is currently frozen.
     * @return If this layers trainable parameters are frozen, then returns true. Otherwise, returns false.
     */
    boolean isFrozen();


    /**
     * Freezes this layers trainable parameters. That is, this layers parameters will NOT be changed
     * during the optimization step of the training loop. If this layer is already frozen, then this method does nothing.
     * <br><br>
     * Note, by default, a {@link TrainableLayer} is not frozen.
     */
    void freeze();


    /**
     * Unfreezes this layers trainable parameters. That is, this layers parameters will be updated during the
     * optimization step of the training loop. If this layer is already unfrozen, then this method does nothing.
     * <br><br>
     * Note, by default, a {@link TrainableLayer} is not frozen.
     */
    void unfreeze();
}
