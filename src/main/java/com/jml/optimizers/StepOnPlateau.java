package com.jml.optimizers;


/**
 * StepOnPlateau is a {@link Scheduler} which "steps" the learning rate whenever the loss stops significantly decreasing
 * for a specified number of epochs.<br><br>
 *
 * The step is computed by a factor. i.e.
 * <pre>
 *      learning_rate = learning_rate * factor
 * </pre>
 *
 *
 * The default step factor is 0.5. <br>
 * The default patience is 2 iterations.<br>
 * The default threshold is 1e-4.
 */
public class StepOnPlateau extends Scheduler {
    // Default hyper-parameter values.
    private final double DEFAULT_FACTOR = 0.5;
    private final int DEFAULT_PATIENCE = 2;
    private final double DEFAULT_THRESHOLD = this.threshold = 1e-4;;

    double factor; // Factor to multiply learning rate by.
    int patience; // Number of epochs to wait before stepping the learning rate

    /* Threshold for measuring a 'significant' change in the loss. If the change in the loss is less than
    the threshold for <patience> number of epochs, then the learning rate will be stepped. */
    double threshold;


    /**
     * Creates a StepOnPlateau {@link Scheduler} with default factor=0.8 and patience=2.
     */
    public StepOnPlateau() {
        factor = DEFAULT_FACTOR;
        patience = DEFAULT_PATIENCE;
        this.threshold = DEFAULT_THRESHOLD;
    }


    /**
     * Creates a StepOnPlateau {@link Scheduler} with specified factor.
     * The patience will default to 2.
     *
     * @param factor Factor to step the learning rate by.
     */
    public StepOnPlateau(double factor) {
        this.factor = factor;
        patience = DEFAULT_PATIENCE;
        this.threshold = DEFAULT_THRESHOLD;
    }


    /**
     * Creates a StepOnPlateau {@link Scheduler} with specified factor and patience.
     *
     * @param factor Factor to step the learning rate by.
     * @param patience Number of epochs to wait before applying the step.
     */
    public StepOnPlateau(double factor, int patience) {
        this.factor = factor;
        this.patience = patience;
        this.threshold = DEFAULT_THRESHOLD;
    }


    /**
     * Creates a StepOnPlateau {@link Scheduler} with specified factor, patience and threshold.
     *
     * @param factor Factor to step the learning rate by.
     * @param patience Number of epochs to wait before applying the step.
     * @param threshold The
     */
    public StepOnPlateau(double factor, int patience, double threshold) {
        this.factor = factor;
        this.patience = patience;
        this.threshold = threshold;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void step() {
        // TODO:
    }
}
