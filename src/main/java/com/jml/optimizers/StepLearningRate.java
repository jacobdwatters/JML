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
    double interval;


    /**
     * Creates a default {@link StepLearningRate} scheduler with factor of 0.8 and an interval of 10.
     */
    public StepLearningRate() {
        this.stepFactor = 0.8;  // Default Step factor
        this.interval = 10; // Default step interval
    }


    /**
     * Creates a {@link StepLearningRate} with specified factor and the default interval of 10.
     *
     * @param stepFactor Factor to multiply the learning rate by every 10 iterations.
     */
    public StepLearningRate(double stepFactor) {
        this.stepFactor=stepFactor;
    }


    /**
     * Creates a {@link StepLearningRate} with specified factor and the interval.
     *
     * @param stepFactor Factor to multiply the learning rate by every interval.
     * @param interval Specified number of optimizer iterations to apply before applying this
     *                 scheduler rule.
     */
    public StepLearningRate(double stepFactor, double interval) {
        this.stepFactor=stepFactor;
        this.interval = interval;
    }


    /**
     * @param optm Optimizer to apply this Scheduler to.
     *
     * return The new learning rate according to the specific Schedulers rules.
     */
    @Override
    public void apply(Optimizer optm) {
        if(optm.iterations!=0 && optm.iterations%interval == 0) { // Then we apply the StepLearningRate optimizer
            optm.learningRate*=stepFactor;
        } // Otherwise, do nothing.
    }
}
