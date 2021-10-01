package com.jml.core.losses;

import com.jml.util.ArrayErrors;
import com.jml.core.Stats;

/**
 * This class contains lambda functions for various loss functions including:
 * <pre>
 *     - {@link #mse mse}: meanNormalize-squared-error.
 * </pre>
 */
public class Loss {

    // A private constructor to hide the implicit constructor.
    private Loss() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * The meanNormalize-squared-error loss function.<br>
     * That is <code>mse = (1/n)sum(x<sub>i</sub> - y<sub>i</sub>)</code>
     * where <code>x<code/> and <code>y<code/> are datasets of length <code>n<code/>,
     * and <code>x<code/> is the actual data and <code>y<code/> is the predicted data.
     */
    static LossFunction mse = (double[] actual, double[] expected) -> {
        ArrayErrors.checkSameLength(actual, expected);

        return Stats.sse(actual, expected)/actual.length;
    };
}
