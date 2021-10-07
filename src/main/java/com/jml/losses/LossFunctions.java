package com.jml.losses;

import com.jml.core.Model;
import linalg.Matrix;

/**
 * This class contains lambda functions for various loss functions including:
 * <pre>
 *     - {@link #sse mse}: sum of squared-errors loss.
 * </pre>
 */
public class LossFunctions {

    // A private constructor to hide the implicit constructor.
    private LossFunctions() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * The sum of squared-errors loss function.<br>
     * That is <code>sse = sum(x<sub>i</sub> - y<sub>i</sub>)<sup>2</sup></code>
     * where <code>x<code/> and <code>y<code/> are datasets of length <code>n<code/>,
     * and <code>x<code/> is the actual data and <code>y<code/> is the predicted data.
     */
    static Function sse = (Matrix w, Matrix X, Matrix y, Model model) -> {
        // TODO: replace X.mult(w) with model.predict(X, w) so that this loss works for any model.
        return X.mult(w).sub(y).T().mult(X.mult(w).sub(y));
    };
}
