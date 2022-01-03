package com.jml.optimizers;


/**
 * TimeDecay is a {@link Scheduler} that decays the learning rate through time. That is, decays the learning rate
 * over epochs. <br><br>
 *
 * The following rule is applied by the TimeDecay {@link Scheduler}:
 * <pre>
 *      learning_rate = learning_rate / (1 + decay*epoch)
 * </pre>
 *
 * By default, <code>decay = initial_learning_rate / total_epochs</code> but a custom value can be specified.
 */
public class TimeDecay extends Scheduler {

    private double decay; // Decay rate.
    private int epochCount; // Counts the number of epochs.


    /**
     * Creates a TimeDecay scheduler with <code>decay = initial_learning_rate / total_epochs</code>.
     * @param totalEpochs Total number of epochs the optimizer will run.
     */
    public TimeDecay(int totalEpochs) {
        decay = optim.learningRate / totalEpochs;
    }


    /**
     * Creates a TimeDecay scheduler with specified decay rate.
     *
     * @param totalEpochs Total number of epochs the optimizer will run.
     * @param decay Decay-rate for this scheduler.
     */
    public TimeDecay(int totalEpochs, double decay) {
        this.decay = decay;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void step() {
        optim.learningRate = optim.learningRate / (1+decay*epochCount);
        epochCount++;
    }
}
