package com.jml.optimizers;

import com.jml.core.Gradient;
import com.jml.core.Model;
import com.jml.losses.Function;

import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * The Stochastic Gradient Descent optimizer. This optimizer minimizes a loss function with respect to the parameters of
 * a model. The loss function is dependent on the features and targets of the training dataset so these must be passed to
 * the optimize method.
 */
public class StochasticGradientDescent extends Optimizer{

    private double threshold = 0.5e-5; // Threshold for stopping optimizer.
    private int maxIterations = 1500;
    private double learningRate = 0.2;
    private Model model;


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


    public StochasticGradientDescent(Model model, double learningRate, int maxIterations, double threshold) {
        this.model = model;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
    }


    /**
     * Applies specified optimizer rule to loss function.
     *
     * @param loss The loss function to optimize.
     * @param X Features of model.
     * @param y Targets of model.
     * @return The computed minimum of the loss function.
     */
    public Matrix optimize(Function loss, Matrix X, Matrix y) {
        List<Double> lossHistory = new ArrayList<>();
        Matrix w = Vector.randn(X.numCols(), 1, false);

        lossHistory.add(loss.compute(w, X, y, this.model).getAsDouble(0, 0));

        int i=0; // To keep track of how many iterations have been completed.

        // TODO: Need the loss function so we can compute loss_history[-1] > threshold as a stopping criteria.
        while(i<maxIterations && lossHistory.get(lossHistory.size()-1) > threshold) {
            // w = w - alpha*grad_SSE(w, x, y)
            w = w.sub(Gradient.compute(w, X, y, loss, model).scalMult(learningRate));
            lossHistory.add(loss.compute(w, X, y, model).getAsDouble(0, 0));
            i++;
        }

        return w;
    }
}
