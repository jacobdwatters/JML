package com.jml.optimizers;

import com.jml.core.Gradient;
import com.jml.linear_models.LinearRegression;
import com.jml.losses.Function;
import linalg.Matrix;

import java.util.ArrayList;
import java.util.List;

public class StochasticGradientDescent {

    private static double threshold = 0.5e-10; // threshold for stopping optimizer.
    private static final int DEFAULT_MAX_ITERATIONS = 1500;


    /**
     * Applies specified optimizer rule to x. Defaults to a max of 1500 iterations.
     *
     * @param X Features of model.
     * @param y Targets of model.
     * @param loss The loss function to optimize.
     * @param alpha Learning rate for the optimization.
     * @return The computed minimum of the loss function.
     */
    public static Matrix optimize(Matrix w, Matrix X, Matrix y, Function loss, double alpha) {
        return optimize(w, X, y, loss, alpha, DEFAULT_MAX_ITERATIONS);
    }


    /**
     * Applies specified optimizer rule to x.
     *
     * @param X Features of model.
     * @param y Targets of model.
     * @param loss The loss function to optimize.
     * @param alpha Learning rate for the optimization.
     * @param maxIterations The maximum iterations to run the optimizer for.
     * @return The computed minimum of the loss function.
     */
    // TODO: Should the loss function be passed in, then the gradient numerically computed? Probably so.
    public static Matrix optimize(Matrix w, Matrix X, Matrix y, Function loss, double alpha, int maxIterations) {

        // TODO: need the loss function passed in to compute this
        List<Double> lossHistory = new ArrayList<>();
        lossHistory.add(loss.compute(w, X, y, new LinearRegression()).getAsDouble(0, 0));

        int i=0; // To keep track of how many iterations have been completed.

        // TODO: Need the loss function so we can compute loss_history[-1] > threshold as a stopping criteria.
        while(i<maxIterations) {
            // TODO: will optimize need to take 'Model model' as a parameter

            // w = w - alpha*grad_SSE(w, x, y)
            w = w.sub(Gradient.compute(w, X, y, loss, new LinearRegression()).scalMult(alpha));
            lossHistory.add(loss.compute(w, X, y, new LinearRegression()).getAsDouble(0, 0));
            i++;
        }

        return w;
    }
}
