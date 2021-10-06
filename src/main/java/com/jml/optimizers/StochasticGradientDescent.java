package com.jml.optimizers;

import com.jml.losses.Function;
import linalg.Matrix;

public class StochasticGradientDescent {

    private double threshold = 0.5e-10; // threshold for stopping optimizer.

    /**
     * Applies specified optimizer rule to x. Defaults to a max of 1500 iterations.
     *
     * @param alpha
     * @param X Features of model.
     * @param y Targets of model.
     * @param lossGrad The loss function to optimize.
     * @return
     */
    public double optimize(int alpha, Matrix X, Matrix y, Function lossGrad) {
        return optimize(alpha, X, y, lossGrad, 1500);
    }


    /**
     * Applies specified optimizer rule to x.
     *
     * @param alpha Learning rate for the optimization.
     * @param X Features of model.
     * @param y Targets of model.
     * @param lossGrad The loss function to optimize.
     * @param maxIterations The maximum iterations to run the optimizer for.
     * @return
     */
    public double optimize(int alpha, Matrix X, Matrix y, Function lossGrad, int maxIterations) {

        // Initial random weights
        Matrix w_start = Matrix.randn(1, X.numCols(), false);


        return 0;
    }



    public static void main(String[] args) {

    }
}
