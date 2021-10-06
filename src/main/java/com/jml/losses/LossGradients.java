package com.jml.losses;

import com.jml.core.Model;
import linalg.Matrix;

public class LossGradients {

    // A private constructor to hide the implicit constructor.
    private LossGradients() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * Gradient of the {@link LossFunctions#sse sse} function.
     */
    static Function sseGrad = (Matrix w, Matrix X, Matrix y, Model model) -> {
        return X.T().mult(X).mult(w).sub(X.T().mult(y));
    };

}
