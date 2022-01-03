package com.jml.optimizers;

/**
 * StepLearningRate is a {@link Scheduler} which "steps" the learning rate at regular intervals during optimization.
 * The step is computed by a factor. That is, every specified number of iterations of the optimizer, the learning rate
 * will be multiplied by this factor.<br><br>
 *
 * The default step factor is 0.8. <br>
 * The default interval is 10 iterations.
 */
public class StepLearningRate extends Scheduler {

    double stepFactor;
    int interval;
    int iterations;


    /**
     * Creates a default {@link StepLearningRate} scheduler with factor of 0.8 and an interval of 10.
     *
     * @param optim {@link Optimizer} to apply learning rate scheduler to.
     */
    public StepLearningRate(Optimizer optim) {
        super.optim = optim;
        this.stepFactor = 0.8;  // Default Step factor
        this.interval = 10; // Default step interval
        this.iterations = 0;
    }


    /**
     * Creates a {@link StepLearningRate} with specified factor and the default interval of 10.
     *
     * @param optim {@link Optimizer} to apply learning rate scheduler to.
     * @param stepFactor Factor to multiply the learning rate by every 10 iterations.
     */
    public StepLearningRate(Optimizer optim, double stepFactor) {
        super.optim = optim;
        this.stepFactor = stepFactor;
    }


    /**
     * Creates a {@link StepLearningRate} with specified factor and the interval.
     *
     * @param optim {@link Optimizer} to apply learning rate scheduler to.
     * @param stepFactor Factor to multiply the learning rate by every interval.
     * @param interval Specified number of optimizer iterations to apply before applying this
     *                 scheduler rule.
     */
    public StepLearningRate(Optimizer optim, double stepFactor, int interval) {
        super.optim = optim;
        this.stepFactor = stepFactor;
        this.interval = interval;
    }


    /**
     * Steps the learning rate scheduler.
     * return The new learning rate according to the specific Schedulers rules.
     */
    @Override
    public void step() {
        if(iterations!=0 && iterations%interval == 0) {
            // Then we apply the StepLearningRate optimizer
            optim.learningRate*=stepFactor;
        } // Otherwise, do nothing.
        iterations++;
    }
}
