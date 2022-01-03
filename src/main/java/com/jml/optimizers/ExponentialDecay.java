package com.jml.optimizers;


/**
 * ExponentialDecay is a {@link Scheduler} that decays the learning rate exponentially.<br><br>
 *
 * The following update rule is applied by the ExponentialDecay {@link Scheduler}.
 * <pre>
 *      learning_rate = initial_learning_rate * exp(-decay*epoch)
 * </pre>
 */
public class ExponentialDecay extends Scheduler {

    private double decay;
    private double initLearningRate;
    private int epochCount = 0;

    /**
     * Creates an ExponentialDecay {@link Scheduler} with specified decay rate.
     *
     * @param decay The decay rate for this scheduler.
     */
    public ExponentialDecay(double decay) {
        this.decay = decay;
        this.initLearningRate = optim.learningRate;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void step() {
        optim.learningRate = this.initLearningRate*Math.exp(-decay*epochCount);
        epochCount++;
    }
}
