package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.core.ModelTypes;
import com.jml.preprocessing.Normalize;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import com.jml.core.Stats;
import linalg.Matrix;
import linalg.Solvers;
import linalg.Vector;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;


/**
 * Model for least squares linear regression of multiple variables by least squares.<br><br>
 *
 * MultipleLinearRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x + ... + b<sub>n</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model.
 */
// TODO: Refactor polynomial regression as a special case of multiple regression
public class MultipleLinearRegression extends Model<double[][], double[]> {
    final String MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION.toString();

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
    public static final String CORRELATION_KEY = "r";

    /**
     * Key for computation of coefficient of determination. <br>
     * The associated value will indicate weather to compute the coefficient of determination after regression.
     */
    public static final String DETERMINATION_KEY = "r2";

    protected int normalization = 0; // Default is no normalization.
    protected double[] coefficients;

    // Details of model in human-readable format.
    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Compiled: No\n" +
                    "Is Trained: No\n"
    );

    /**
     * Constructs model and prepares for training using the given parameters.
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
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * <pre>
     *  - Normalization:
     *      <"normalize", 0> - Default. No normalization is used.
     *      <"normalize", 1> - Normalizes data by subtracting meanNormalize and dividing by the L2-norm before applying regression.
     * </pre>
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
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
        }

        isCompiled = true; // Set the compiled flag to true.
        buildDetails(); // Build the details of the model.
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @param args     A hashtable containing additional arguments in the form <name, value>.
     * @return Returns details of the fitting / training process.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    @Override
    public ModelBucket fit(double[][] features, double[] targets, Map<String, Double> args) {

        Map<String, Object> results = new HashMap<>();
        boolean computeCorrelation = false, computeDetermination = false;

        if(!isCompiled) {
            throw new IllegalStateException("Model must be compiled before it can be fit.");
        }

        if(!Objects.isNull(args) && !args.isEmpty()) { // Check for various optional arguments
            if(args.containsKey(CORRELATION_KEY)) {
                computeCorrelation = true;
            }
            if(args.containsKey(DETERMINATION_KEY)) {
                computeDetermination = true;
            }
        }

        if(normalization==1) { // Then use l2 normalization.
            features = Normalize.l2(features);
        }

        Matrix y = (new Vector(targets)).toMatrix();
        Matrix V = new Matrix(features);
        Matrix ones = Matrix.ones(features.length, 1);
        V = ones.augment(V);

        Matrix VT = V.T();
        Matrix A = VT.mult(V);
        Vector b = VT.mult(y).toVector();

        if(A.isSingular()) { // Then we can not explicitly solve Ax=b for a single solution.
            throw new IllegalArgumentException("The data resulted in an equation with a singular matrix. " +
                    "Singular matrices are not supported. Use the MultipleLinearRegressionSGD model instead.");
        }

        coefficients = Solvers.solve(A, b).T().getValuesAsDouble()[0]; // Compute the model parameters
        results.put("coefficients", coefficients);
        isFit = true;

        if(computeCorrelation) {
            results.put("r", Stats.correlation(targets, this.predict(features)));
        }
        if(computeDetermination) {
            results.put("r2", Stats.determination(targets, this.predict(features)));
        }

        buildDetails(); // Build the details of the model.

        return new ModelBucket(results);
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return Returns details of the fitting / training process in a {@link ModelBucket}.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public ModelBucket fit(double[][] features, double[] targets) {
        return fit(features, targets, null);
    }


    /**
     * Uses fitted/trained model to make prediction on single feature.
     *
     * @param features The features to make predictions on.
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public double[] predict(double[][] features) {
        if(!isFit || !isCompiled) {
            throw new IllegalStateException("Model must be compiled and fit before predictions can be made.");
        }

        double[] predictions = new double[features.length];

        for(int i=0; i<features.length; i++) {

            for (int j = 1; j < coefficients.length; j++) {
                predictions[i] += coefficients[j]*features[i][j-1];
            }

            predictions[i] += coefficients[0];
        }

        return predictions;
    }


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    @Override
    public void saveModel(String filePath) {
        Block[] blockList;

        if(!isFit) {
            throw new IllegalStateException("Model must be fit before it can be saved.");
        }
        if(!filePath.substring(filePath.length()-4,filePath.length()).equals(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        blockList = new Block[3];

        // Construct the blocks for the model file.
        blockList[0] = new Block(LinearModelTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(LinearModelTags.COEFFICIENTS.toString(), ArrayUtils.asString(this.coefficients));
        blockList[2] = new Block(LinearModelTags.NORMALIZE.toString(), Integer.toString(normalization));

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }


    // Construct details of model
    protected void buildDetails() {
        details = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE+ "\n" +
                        "Is Compiled: " + (isCompiled ? "Yes" : "No") + "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );


        if(isCompiled) {
            details.append("Normalization: " + (normalization==1 ? "Yes" : "No") + "\n");
        }


        if(isFit && coefficients!=null) {
            details.append("Coefficients: ");
            details.append(ArrayUtils.asString(coefficients));
            details.append("\nHyperplane: y = " + coefficients[0] + " + ");

            for(int i=1; i<coefficients.length; i++) {
                details.append(coefficients[i] + "x_" + i);

                if(i<coefficients.length-1) {
                    details.append(" + ");
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
        return details.toString();
    }
}
