package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.core.ModelTypes;
import com.jml.losses.LossFunctions;
import com.jml.optimizers.StochasticGradientDescent;
import linalg.Matrix;
import linalg.Vector;

import java.util.HashMap;
import java.util.Map;

public class MultipleLinearRegressionSGD extends MultipleLinearRegression {

    public MultipleLinearRegressionSGD() {
        super.updateModelType(ModelTypes.MULTIPLE_LINEAR_REGRESSION_SGD.toString());
    }

    public ModelBucket fit(double[][] features, double[] targets, Map<String, Double> args) {
        Map<String, Object> results = new HashMap<>();

        // Convert features and targets to matrix representations.
        Matrix X = Matrix.ones(features.length, 1).augment(new Matrix(features));
        Matrix y = new Vector(targets);

        w = Matrix.randn(X.numCols(), 1, false); // Set initial model weights to be random.
        w = StochasticGradientDescent.optimize(w, X, y, LossFunctions.sse, 0.05, 3000);

        super.coefficients = w.T().getValuesAsDouble()[0];
        results.put("coefficients", super.coefficients);

        super.isFit=true;
        super.buildDetails();

        return new ModelBucket(results);
    }
}
