package com.jml.core;

import com.jml.losses.Function;
import linalg.Matrix;

/**
 * Allows for computation of the gradient of a function. Functions are assumed to be dependent on a matrix w.
 */
public class Gradient {

    private static final double eps = 0.5e-8;

    private Gradient() {
        throw new IllegalStateException("Utility method cannot be instantiated.");
    }


    /**
     * Numerically computes the gradient of a loss function for a specified model. This is computed
     * using the three point centered difference formula. i.e. f'(x) ~ (f(x+h)-f(x-h)) / 2h
     *
     * @param w Parameters of the model.
     * @param X Features of the dataset.
     * @param y Targets of the dataset.
     * @param F Function to compute gradient of.
     * @param model Model to use for gradient computation.
     * @return Gradient with respect to w for the specified function.
     */
    // TODO: Only take function,
    public static Matrix compute(Matrix w, Matrix X, Matrix y, Function F, Model model) {

        Matrix grad = new Matrix(w.shape());
        Matrix yPred1, yPred2; // Predictions using weights adjusted by +eps and -eps respectively.

        // A diagonal matrix containing the value of h along the diagonal.
        Matrix H = Matrix.I(w.numRows()).scalMult(eps);
        Matrix functionValue = F.compute(y, model.predict(X, w));

        for(int i=0; i<w.numRows(); i++) { // Compute partial derivative of F with respect to each w_i in w
            yPred1 = F.compute(y, model.predict(X, w.add(H.getColAsVector(i))));
            yPred2 = F.compute(y, model.predict(X, w.sub(H.getColAsVector(i))));

            Matrix partial = yPred1.sub(yPred2).scalDiv(2*eps);

            // Set the gradient at the given index to be the computed partial derivative.
            grad.set(partial.getAsDouble(0, 0), i, 0);
        }

        return grad;
    }
}
