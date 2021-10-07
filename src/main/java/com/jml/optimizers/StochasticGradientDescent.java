package com.jml.optimizers;

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
     * @param lossGrad The gradient of the loss function to optimize.
     * @param alpha Learning rate
     * @return
     */
    public static Matrix optimize(Matrix w, Matrix X, Matrix y, Function lossGrad, double alpha) {
        return optimize(w, X, y, lossGrad, alpha, DEFAULT_MAX_ITERATIONS);
    }


    /**
     * Applies specified optimizer rule to x.
     *
     *
     * @param X Features of model.
     * @param y Targets of model.
     * @param lossGrad The gradient of the loss function to optimize.
     * @param alpha Learning rate for the optimization.
     * @param maxIterations The maximum iterations to run the optimizer for.
     * @return
     */
    // TODO: Should the loss function be passed in, then the gradient numerically computed? Probably so.
    public static Matrix optimize(Matrix w, Matrix X, Matrix y, Function lossGrad, double alpha, int maxIterations) {

        // TODO: need the loss function passed in to compute this
        List<Double> lossHistory = new ArrayList<>();
        int i=0;

        // TODO: Need the loss function so we can compute loss_history[-1] > threshold as a stopping criteria.
        while(i<maxIterations) {
            // TODO: will optimize need to take 'Model model' as a parameter
            // w = w - alpha*grad_SSE(w, x, y)
            w = w.sub(lossGrad.compute(w, X, y, new LinearRegression()).scalMult(alpha));
            i++;
        }

        return w;
    }
}
