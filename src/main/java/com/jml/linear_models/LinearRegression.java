package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.linalg.Matrix;
import com.jml.linalg.Solvers;
import com.jml.linalg.Vector;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;

import java.util.Map;
import java.util.Objects;


/**
 * Model for ordinary least squares linear regression of one variable.<br><br>
 *
 * LinearRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is solved explicitly.
 */
public class LinearRegression extends PolynomialRegression {
    public final String MODEL_TYPE = ModelTypes.LINEAR_REGRESSION.toString();
    private String details = "Model Details\n" +
            "----------------------------\n" +
            "Model Type: " + this.MODEL_TYPE+ "\n" +
            "Is Compiled: No\n" +
            "Is Trained: No\n";

    /**
     * Constructs model and prepares for training using default parameters. i.e. no normalization will be used.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile() {
        super.compile();
        buildDetails();
    }


    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * Valid additional args. All others will be ignored.
     * <pre>
     *  - Normalization:
     *      <"normalize", 0> - Default. No normalization is used.
     *      <"normalize", 1> - Normalizes data by subtracting mean and dividing by the L2-norm before applying regression.
     * <pre/>
     *
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @throws IllegalArgumentException If values in <code>args</code> are invalid of a specified key. Unspecified keys will simply be
     * ignored and will NOT throw error.
     */
    @Override
    public void compile(Map<String, Double> args) {
        if(!Objects.isNull(args) && !args.isEmpty()) {
            if(args.containsKey(LinearRegression.DEGREE_KEY)) { // Ensure no degree is passed to super class.
                args.remove(LinearRegression.DEGREE_KEY);
            }
        }

        super.compile(args);
        buildDetails();
    }


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
    public double[][] fit(double[] features, double[] targets) {
        double[][] result = super.fit(features, targets, null);
        buildDetails();
        return result;
    }


    /**
     * Fits the model with the given features and targets.
     *
     * Valid additional args. All others will be ignored.
     * <pre>
     *  - R value (goodness of fit):
     *      <"fit", 0> - Default. No R value is returned.
     *      <"fit", 1> - R value of model will be calculated and returned.
     * <pre/>
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @param args     A hashtable containing additional arguments in the form <name, value>.
     * @return A 2D array containing the following on a row: <br>
     *  - The coefficients of the line from lowest to highest degree.
     *  - The R value (goodness of fit) if indicated in args.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    // TODO: Add ability to get R value.
    @Override
    public double[][] fit(double[] features, double[] targets, Map<String, Double> args) {
        double[][] result = super.fit(features, targets, args);
        buildDetails();
        return result;
    }


    // Construct details of model
    @Override
    protected void buildDetails() {
        details ="Model Details\n" +
                "----------------------------\n" +
                "Model Type: " + this.MODEL_TYPE+ "\n" +
                "Is Compiled: " + (isCompiled ? "Yes" : "No") + "\n" +
                "Is Trained: " + (isFit ? "Yes" : "No") + "\n";

        if(isCompiled) {
            details += "Normalization: " + (normalization==1 ? "Yes" : "No") + "\n";
        }

        if(isFit && coefficients!=null) {
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
