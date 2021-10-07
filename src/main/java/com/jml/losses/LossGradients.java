package com.jml.losses;

import com.jml.core.Model;
import linalg.Matrix;

public class LossGradients {

    // A private constructor to hide the implicit constructor.
    private LossGradients() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }
    static double h = 0.5e-8;


    /**
     * Gradient of the {@link LossFunctions#sse sse} function for a Linear, MultiLinear, or polynomial regression model.
     */
    public static Function sseLinRegGrad = (Matrix w, Matrix X, Matrix y, Model model) -> {
        return (X.T()).mult(X).mult(w).sub(X.T().mult(y));
    };



    public static Function sseGrad = (Matrix w, Matrix X, Matrix y, Model model) -> {

        Matrix grad = new Matrix(w.shape());

        // A diagonal matrix containing the value of h along the diagonal.
        Matrix H = Matrix.I(w.numRows()).scalMult(h);

        for(int i=0; i<w.numRows(); i++) { // Compute partial derivative for each w_i in w

            Matrix partial = LossFunctions.sse.compute(
                    w.add(H.getColAsVector(i)), X, y, model).sub(
                        LossFunctions.sse.compute(w, X, y, model)
                    ).scalDiv(h);

            // Set the gradient at the given index to be the computed partial derivative.
            grad.set(partial.getAsDouble(0, 0), i, 0);
        }

        System.out.println("\n\ngrad:\n" + grad.toString() + "\n\n");

        return grad;
    };

}
