package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;

import com.jml.util.Stats;
import linalg.Matrix;
import linalg.Solvers;
import linalg.Vector;

import java.lang.Math;
import java.util.Map;
import java.util.Objects;


/**
 * Model for least squares linear regression of polynomials.<br><br>
 *
 * PolynomialRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x  + b<sub>2</sub>x<sup>2</sup> + ... +
 * b<sub>n</sub>x<sup>n</sup> to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is solved explicitly.
 */
public class PolynomialRegression extends Model<double[], double[]> {
    final String MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION.toString();

    protected boolean isFit = false, isCompiled = false;

    /**
     * Key for the degree of the polynomial. <br>
     * The associated value will be the degree for the polynomial used in regression.
     */
    public static final String DEGREE_KEY = "degree";

    /**
     * Key for use of normalization before regression. <br>
     * The associated value will indicate weather to normalize the features before regression.
     */
    public static final String NORMALIZE_KEY = "normalize";

    /**
     * Key for computation of correlation coefficient. <br>
     * The associated value will indicate weather to compute the correlation coefficient after regression.
     */
    public static final String CORRELATION_KEY = "R";

    /**
     * Key for computation of coefficient of determination. <br>
     * The associated value will indicate weather to compute the coefficient of determination after regression.
     */
    public static final String DETERMINATION_KEY = "R2";

    protected int degree = 1; // Defaults to simple linear regression.
    protected int normalization = 0; // Default is no normalization.
    protected double[] coefficients;
    private String details = "Model Details\n" +
            "----------------------------\n" +
            "Model Type: " + this.MODEL_TYPE+ "\n" +
            "Is Compiled: No\n" +
            "Is Trained: No\n";

    /**
     * Constructs model and prepares for training using default parameters. i.e. the degree of the polynomial will be 1,
     * and no normalization will be used.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile() {
        compile(null);
    }


    /**
     * Constructs model and prepares for training using the given parameters.
     *
     * Valid additional args. All others will be ignored.
     * <pre>
     *  - Degree of polynomial to fit:
     *      <"degree", n> - where n is the integer degree of polynomial to fit. n=1 is the default value.
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
        if(!Objects.isNull(args) && !args.isEmpty()) { // Then args is not null and is not empty
            if(args.containsKey(NORMALIZE_KEY)) {
                double value = args.get(NORMALIZE_KEY);

                if(!(value == 0.0 || value == 1.0)) {
                    throw new IllegalArgumentException("Normalization must be 0 or 1 but got " + value);
                } else {
                    this.normalization = (int) value;
                }
            }
            if(args.containsKey(DEGREE_KEY)) {
                double value = args.get(DEGREE_KEY);
                if(value != (int) value) { // Then value is not an integer
                    throw new IllegalArgumentException("Degree must be integer but got " + value);
                } else if(value <= 0) {
                    throw new IllegalArgumentException("Degree must greater than 0 but got " + value);
                } else {
                    this.degree = (int) value;
                }
            }
        }

        isCompiled = true; // Set the compiled flag to true.
        buildDetails(); // Build the details of the model.
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
     * @return A 2D array containing the following on a row in the following order: <br>
     *  - The coefficients of the polynomial from lowest to highest degree.
     *  - The R value (correlation coefficient, i.e. the amount of correlation) if indicated in args.
     *  - The R^2 value (coefficient of determination, i.e. goodness of fit) if indicated in args.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    // TODO: should this return a map instead?
    @Override
    public double[][] fit(double[] features, double[] targets, Map<String, Double> args) {
        if(!isCompiled) {
            throw new IllegalStateException("Model must be compiled before it can be fit.");
        }

        int resultRows = 1;
        boolean computeCorrelation = false, computeDetermination = false;



        if(!Objects.isNull(args) && !args.isEmpty()) { // Check for various optional arguments
            if(args.containsKey(CORRELATION_KEY)) {
                computeCorrelation = true;
                resultRows++;
            }
            if(args.containsKey(DETERMINATION_KEY)) {
                computeDetermination = true;
                resultRows++;
            }
        }

        isFit = true;
        double[][] results = new double[resultRows][];

        Vector x = new Vector(features);
        Matrix y = (new Vector(targets)).toMatrix();
        Matrix V = Matrix.van(x, degree+1);
        Matrix VT = V.T();

        Matrix A = VT.mult(V);
        Vector b = VT.mult(y).toVector();
        coefficients = Solvers.solve(A, b).T().getValuesAsDouble()[0];
        results[0] = coefficients;


        /* TODO: The return value should almost certainly be a map. To guarantee that things are in the correct order
            For an array, we need to check every case except for none. So for n arguments we must check 2^n-1 cases.
            This is clearly not practical for several arguments. So we should use a map instead.
         */
        if(computeCorrelation && computeDetermination) {
            results[1] = new double[]{Stats.correlation(features, this.predict(targets))};
            results[2] = new double[]{Stats.determination(features, this.predict(targets))};
        } else if(computeCorrelation) {
            results[1] =  new double[]{Stats.correlation(features, this.predict(targets))};
        } else if(computeDetermination) {
            results[1] = new double[]{Stats.determination(features, this.predict(targets))};
        }

        buildDetails(); // Build the details of the model.

        return results;
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return A 2D array containing the following on a row: <br>
     *  - The coefficients of the polynomial from lowest to highest degree.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public double[][] fit(double[] features, double[] targets) {
        return fit(features, targets, null);
    }


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     * the specification when the model was compiled.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     */
    public double[] predict(double[] features) {
        if(!isFit || !isCompiled) {
            throw new IllegalStateException("Model must be compiled and fit before predictions can be made.");
        }

        double[] predictions = new double[features.length];
        int position = 0;

        for(double feature : features) { // For each feature, compute the prediction.
            for (int j = coefficients.length - 1; j >= 0; j--) {
                predictions[position] += coefficients[j] * Math.pow(feature, j);
            }
            position++;
        }

        return predictions;
    }


    /**
     * Saves a trained model to the specified file path including the name of the file.
     * File path must include the extension .mdl.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    @Override
    public void saveModel(String filePath) {
        String linOrPolyType;
        Block[] blockList;

        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }
        if(!filePath.substring(filePath.length()-4,filePath.length()).equals(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        if(this instanceof LinearRegression) {
            blockList = new Block[3];
            linOrPolyType = ModelTypes.LINEAR_REGRESSION.toString();
        } else {
            blockList = new Block[4];
            linOrPolyType = this.MODEL_TYPE;
            blockList[3] = new Block(LinearModelTags.DEGREE.toString(), Integer.toString(degree));
        }

        // Construct the blocks for the model file.
        blockList[0] = new Block(LinearModelTags.MODEL_TYPE.toString(), linOrPolyType);
        blockList[1] = new Block(LinearModelTags.COEFFICIENTS.toString(), ArrayUtils.asString(this.coefficients));
        blockList[2] = new Block(LinearModelTags.NORMALIZE.toString(), Integer.toString(normalization));

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }


    // Construct details of model
    protected void buildDetails() {
        details ="Model Details\n" +
                "----------------------------\n" +
                "Model Type: " + this.MODEL_TYPE+ "\n" +
                "Is Compiled: " + (isCompiled ? "Yes" : "No") + "\n" +
                "Is Trained: " + (isFit ? "Yes" : "No") + "\n";


        if(isCompiled) {
            details += "Normalization: " + (normalization==1 ? "Yes" : "No") + "\n";
            details += "Polynomial Degree: " + degree + "\n";
        }

        if(isFit && coefficients!=null) {
            details += "Coefficients (low->high): ";
            details += ArrayUtils.asString(coefficients);
            details += "\nPolynomial: y = " + coefficients[0] + " + " + coefficients[1] + "x + ";

            for(int i=2; i<coefficients.length; i++) {
                details += coefficients[i] + "x^" + i;

                if(i<coefficients.length-1) {
                    details += " + ";
                }
            }
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
