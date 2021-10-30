package com.jml.optimizers;

import com.jml.core.Gradient;
import com.jml.core.Model;
import com.jml.losses.Function;

import linalg.Matrix;
import linalg.Vector;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

/**
 * The Stochastic Gradient Descent optimizer. This optimizer minimizes a loss function with respect to the parameters of
 * a model. The loss function is dependent on the features and targets of the training dataset so these must be passed to
 * the optimize method.
 */
public class StochasticGradientDescent extends Optimizer {

    // TODO: Add checks to ensure that maxIterations, learningRate, and threshold are non-negative.

    /**
     * Creates a Stochastic Gradient Descent optimizer to minimize the specified loss function for a model.
     *
     * @param model Model to apply optimization to.
     */
    public StochasticGradientDescent(Model model) {
        this.model = model;
    }


    /**
     * Creates a Stochastic Gradient Descent optimizer to minimize the specified loss function for a model.
     *
     * @param model Model to apply optimization to.
     * @param learningRate Learning rate of the Stochastic Gradient Descent optimizer.
     */
    public StochasticGradientDescent(Model model, double learningRate) {
        this.model = model;
        this.learningRate = learningRate;
    }


    /**
     * Creates a Stochastic Gradient Descent optimizer to minimize the specified loss function for a model.
     *
     * @param model Model to apply optimization to.
     * @param learningRate Learning rate of the Stochastic Gradient Descent optimizer.
     * @param maxIterations Maximum iterations to
     */
    public StochasticGradientDescent(Model model, double learningRate, int maxIterations) {
        this.model = model;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
    }


    /**
     * Creates a Stochastic Gradient Descent optimizer to minimize the specified loss function for a model.
     *
     * @param model Model to apply optimization to.
     * @param learningRate Learning rate of the Stochastic Gradient Descent optimizer.
     * @param maxIterations Maximum iterations to
     * @param threshold Threshold for stopping gradient descent. If the loss is less than this value, gradient descent
     *                  will stop early.
     */
    public StochasticGradientDescent(Model model, double learningRate, int maxIterations, double threshold) {
        this.model = model;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
    }


    /**
     * Gets the loss history from the optimizer.
     *
     * @return The loss of every iteration stored in a List.
     */
    @Override
    public List<Double> getLossHist() {
        return this.lossHistory;
    }


    /**
     * Applies specified optimizer rule to function to compute minimum.
     *
     * @param function The function to optimize.
     * @param X Features of model.
     * @param y Targets of model.
     * @return The computed minimum of the function.
     */
    public Matrix optimize(Function function, Matrix X, Matrix y) {

        Matrix w = Vector.randn(X.numCols(), 1, false); // Initial weights.
        Matrix yPred = this.model.predict(X, w);

        lossHistory.add(function.compute(y, yPred).getAsDouble(0, 0));
        iterations = 0; // Ensure iterations is set to zero before applying optimizer.

        while(iterations<maxIterations && lossHistory.get(lossHistory.size()-1) > threshold) {
            // w = w - alpha*grad_SSE(w, x, y)
            w = w.sub(Gradient.compute(w, X, y, function, model).scalMult(learningRate));
            yPred = this.model.predict(X, w);
            lossHistory.add(function.compute(y,yPred).getAsDouble(0, 0));

            if(scheduler!=null) {
                scheduler.apply(this);
            }

            iterations++;
        }

        return w;
    }
}
