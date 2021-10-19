package com.jml.losses;

import com.jml.core.Model;
import linalg.Matrix;

public interface Function {

    /**
     * Evaluates a function which takes features and targets for a model and weights as parameters.
     *
     * @param weights Weights for the model.
     * @param X Features of the model.
     * @param y Targets of the model.
     * @param model Model of interest.
     * @return The loss between the expected and actual data values.
     */
    Matrix compute(Matrix weights, Matrix X, Matrix y, Model model);
}


