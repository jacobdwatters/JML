package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.util.ArrayUtils;
import linalg.Matrix;
import linalg.Solvers;
import linalg.Vector;

import java.util.HashMap;
import java.util.Objects;


/**
 * Model for least squares linear regression of polynomials.<br><br>
 *
 * PolynomialRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x  + b<sub>2</sub>x<sup>2</sup> + ... +
 * b<sub>n</sub>x<sup>n</sup> to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is using stochastic gradient descent.
 */
public class PolynomialRegression extends Model<double[], double[]> {
    public final String MODEL_TYPE = "Polynomaial Regression";
    private final String DEGREE = "degree";
    private final String NORMALIZE = "normalize";
    private int degree = 1; // Defaults to simple linear regression.
    private int normalization = 0; // Default is no normalization.
    private double[] coefficients;

    /**
     * Constructs model and prepares for training using default parameters. i.e. the degree of the polynomial will be 1,
     * and no normalization will be used.
     *
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile() {
        compile(new HashMap<String, Double>());
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
    public void compile(HashMap<String, Double> args) {
        if(!Objects.isNull(args) && !args.isEmpty()) { // Then args is not null and is not empty
            if(args.containsKey(NORMALIZE)) {
                double value = args.get(NORMALIZE);

                if(value != 0 || value != 1) {
                    throw new IllegalArgumentException("Normalization must be 0 or 1 but got " + value);
                } else {
                    this.normalization = (int) value;
                }
            }
            if(args.containsKey(DEGREE)) {
                double value = args.get(DEGREE);
                if(value != (int) value) { // Then value is not an integer
                    throw new IllegalArgumentException("Degree must be integer but got " + value);
                } else {
                    this.degree = (int) value;
                }
            }
        }
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
     *  - The coefficients of the polynomial from highest to lowest degree.
     *  - The R value (goodness of fit) if indicated in args.
     * @throws IllegalArgumentException Can be thrown for the following reasons<br>
     *                                  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *                                  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    @Override
    public double[][] fit(double[] features, double[] targets, HashMap<String, Double> args) {
        double[][] results = new double[1][];
        Vector x = new Vector(features);
        Matrix y = (new Vector(targets)).toMatrix();
        Matrix V = Matrix.van(x, degree+1);
        Matrix VT = V.T();

        Matrix A = VT.mult(V);
        Vector b = VT.mult(y).toVector();
        coefficients = Solvers.solve(A, b).T().getValuesAsDouble()[0];
        results[0] = coefficients;

        return results;
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return A 2D array containing the following on a row: <br>
     *  - The coefficients of the polynomial from highest to lowest degree.
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
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * Saves a trained model to the specified file path.
     *
     * @param filePath File path, including extension, to save fitted / trained model to.
     */
    @Override
    public void saveModel(String filePath) {
        // TODO: Auto-generated method stub
    }


    /**
     * Prints details of model to the standard output.
     */
    @Override
    public void printDetails() {
        String details = this.MODEL_TYPE+"\n";

        for(int i=0; i<this.MODEL_TYPE.length(); i++) {
            details += "-";
        }
        details += "\nCoefficients (high->low): ";
        details += ArrayUtils.asString(coefficients);

        System.out.print(details);
        // TODO: Auto-generated method stub
    }


    public static void main(String[] args) {
        double[] features = {1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        double[] targets = {44.6, 57.8, 49.9, 61.3, 49.6, 61.8, 49.0, 44.7, 59.2, 53.9, 46.5, 54.7, 50.3, 51.2, 45.7};

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 6.0);

        Model m = new PolynomialRegression();
        m.compile(arguments);
        double[][] c  = m.fit(features, targets);

        System.out.println("\n\n\n");
        m.printDetails();
        System.out.println("\n\n\n");
    }
}
