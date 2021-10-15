package com.jml.optimizers;

import com.jml.core.Model;
import com.jml.losses.Function;
import linalg.Matrix;

import java.util.ArrayList;
import java.util.List;


/**
 * An optimizer which minimizes a specified function numerically.<br><br>
 *
 * The function is assumed to depend on two matrices X
 * and y. For instance a {@link com.jml.losses.LossFunctions loss function} depends on two matrices X and y where y is
 * the expected values and a model making predictions on X are the estimated values.
 */
public abstract class Optimizer {

    double threshold = 0.5e-5; // Threshold for stopping optimizer before maxIterations is reached.
    int maxIterations = 1500; // Maximum number of iterations to run optimizer before quiting.
    int iterations; // Tracks how manny iterations the optimizer has run for.
    double learningRate = 0.2; // Learning rate of the optimizer.
    Model model; // Model which the optimizer is working on. This is needed since loss functions depends on a model.
    Scheduler scheduler; // Learning rate scheduler rule to apply during optimization. If this is left as null, then no rule will be applied.
    List<Double> lossHistory = new ArrayList<>(); // Tracks loss per iteration of the optimizer.


    /**
     * Gets the loss history from the optimizer.
     *
     * @return The loss of every iteration stored in a List. If the optimizer has not been applied, returns null.
     */
    public abstract List<Double> getLossHist();

    /**
     * Applies specified optimizer rule to loss function.
     *
     * @param function The loss function to optimize.
     * @param X Features of model.
     * @param y Targets of model.
     * @return The computed minimum of the loss function.
     */
    public abstract Matrix optimize(Function function, Matrix X, Matrix y);


    /**
     * Sets the learning rate scheduler for the optimizer.
     * @param scheduler The scheduler for the optimizer.
     */
    public void setScheduler(Scheduler scheduler) {
        this.scheduler = scheduler;
    }
}
