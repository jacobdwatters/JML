package com.jml.linear_models;

import com.jml.core.Model;
import java.util.HashMap;


/**
 * Model for ordinary least squares linear regression of one variable.<br><br>
 *
 * LinearRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is solved explicitly.
 */
public class LinearRegression extends Model<double[][], double[]> {

    // Used as regressor since simple linear regression is a special case of polynomial regression.
    private PolynomialRegression polyRegress = new PolynomialRegression();

    /**
     * Constructs model and prepares for training using the given parameters.<br>
     * If you would like to add additional arguments see {@link #compile(HashMap) compile(HashMap)}.
     *
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
     * Valid additional arguments.
     * <pre>
     *  - Normalization:
     *      <"normalize", 0> - Default. No normalization is used.
     *      <"normalize", 1> - Normalizes data using min-max scaling.
     * <pre/>
     *
     * If you don't want to add additional arguments consider using {@link #compile() compile()} or pass null for args.
     *
     * @param args A hashtable containing additional arguments in the form <name, value>.
     * @throws IllegalArgumentException If key, value pairs in <code>args</code> are unspecified or invalid arguments.
     */
    @Override
    public void compile(HashMap<String, Double> args) {
        if(args.containsKey("degree")) {
            throw new IllegalArgumentException("Can not pass degree as an argument for " + this.getClass());
        }

        polyRegress.compile(args);
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @param args     A hashtable containing additional arguments in the form <name, value>.
     * @return Returns details of the fitting / training process.
     * @throws IllegalArgumentException
     * Can be thrown for the following reasons<br>
     *  - If key, value pairs in <code>args</code> are unspecified or invalid arguments. <br>
     *  - If the features and targets are not correctly sized per the specification when the model was
     *                                  compiled.
     */
    @Override
    public double[][] fit(double[][] features, double[] targets, HashMap<String, Double> args) {
        // TODO: Auto-generated method stub
        return new double[0][];
    }


    /**
     * Fits or trains the model with the given features and targets.
     *
     * @param features The features of the training set.
     * @param targets  The targets of the training set.
     * @return - Returns details of the fitting / training process.
     * @throws IllegalArgumentException Thrown if the features and targets are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public double[][] fit(double[][] features, double[] targets) {
        // TODO: Auto-generated method stub
        return new double[0][];
    }


    /**
     * Uses fitted/trained model to make predictions on features.
     *
     * @param features Features to make predictions on.
     * @return The models predicted labels.
     * @throws IllegalArgumentException Thrown if the features are not correctly sized per
     *                                  the specification when the model was compiled.
     */
    @Override
    public double[] predict(double[][] features) {
        // TODO: Auto-generated method stub
        return new double[0];
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
        // TODO: Auto-generated method stub
    }

    public static void main(String[] args) {
        Model<?, ?> l = new LinearRegression();
        HashMap<String, Double> args_ = new HashMap<String, Double>();


        l.compile(args_);

    }
}
