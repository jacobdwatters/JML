package com.jml.linear_models;

import com.jml.core.ModelTypes;
import com.jml.util.ArrayUtils;


/**
 * Model for ordinary least squares linear regression of one variable.<br><br>
 *
 * LinearRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is solved explicitly.
 */
public class LinearRegression extends PolynomialRegression {
    String MODEL_TYPE = ModelTypes.LINEAR_REGRESSION.toString();
    private String details = "Model Details\n" +
            "----------------------------\n" +
            "Model Type: " + this.MODEL_TYPE+ "\n" +
            "Is Trained: No\n";


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return A 2D array containing the following on a row: <br>
     *  - The coefficients of the line from lowest to highest degree.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public LinearRegression fit(double[] features, double[] targets) {
        super.fit(features, targets);
        buildDetails();
        return this;
    }


    // Construct details of model
    @Override
    protected void buildDetails() {
        details ="Model Details\n" +
                "----------------------------\n" +
                "Model Type: " + this.MODEL_TYPE+ "\n" +
                "Is Trained: " + (isFit ? "Yes" : "No") + "\n";

        if(isFit && w!=null) {
            coefficients = w.T().getValuesAsDouble()[0];
            details += "Coefficients (low->high): ";
            details += ArrayUtils.asString(coefficients);
            details += "\nLine: y = " + coefficients[0] + " + " + coefficients[1] + "x";
        }
    }


    /**
     * Forms a string of the important aspects of the model.<br>
     * same as {@link #toString()}
     *
     * @return Details of model as string.
     */
    @Override
    public String getDetails() {
        return this.toString();
    }


    /**
     * Forms a string of the important aspects of the model.
     *
     * @return String representation of model.
     */
    @Override
    public String toString() {
        return details;
    }
}
