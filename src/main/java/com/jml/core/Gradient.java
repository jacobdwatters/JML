package com.jml.core;

import com.jml.losses.Function;
import com.jml.losses.LossFunctions;
import linalg.Matrix;

public class Gradient {

    private static double h = 0.5e-8;

    private Gradient() {
        throw new IllegalStateException("Utility method cannot be instantiated.");
    }


    /**
     * Numerically computes the gradient of a loss function for a specified model.
     *
     * @param w Parameters of the model.
     * @param X Features of the dataset.
     * @param y Targets of the dataset.
     * @param F Function to compute gradient of.
     * @param model Model to use for gradient computation.
     * @return Gradient with respect to w for the specified function.
     */
    public static Matrix compute(Matrix w, Matrix X, Matrix y, Function F, Model model) {

        Matrix grad = new Matrix(w.shape());

        // A diagonal matrix containing the value of h along the diagonal.
        Matrix H = Matrix.I(w.numRows()).scalMult(h);
        Matrix functionValue = F.compute(w, X, y, model);

        for(int i=0; i<w.numRows(); i++) { // Compute partial derivative for each w_i in w

            Matrix partial = F.compute(
                    w.add(H.getColAsVector(i)), X, y, model).sub(
                    functionValue
            ).scalDiv(h);

            // Set the gradient at the given index to be the computed partial derivative.
            grad.set(partial.getAsDouble(0, 0), i, 0);
        }

        return grad;
    }
}
