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
}
