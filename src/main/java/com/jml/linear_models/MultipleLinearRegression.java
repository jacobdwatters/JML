package com.jml.linear_models;

import com.jml.core.Block;
import com.jml.core.Model;
import com.jml.core.ModelTypes;
import com.jml.util.ArrayUtils;
import com.jml.util.FileManager;
import linalg.Matrix;
import linalg.Solvers;
import linalg.Vector;


/**
 * Model for linear regression of multiple variables by least squares.<br><br>
 *
 * MultipleLinearRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + ... + b<sub>n</sub>x<sub>n</sub>  to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model.
 */
public class MultipleLinearRegression extends Model<double[][], double[]> {
    String MODEL_TYPE = ModelTypes.MULTIPLE_LINEAR_REGRESSION.toString();

    protected boolean isFit = false;
    protected double[] coefficients;

    // Weights of the model.
    protected Matrix w;

    // Details of model in human-readable format.
    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );


    /**
     * Creates a multivariate least squares regression model
     */
    public MultipleLinearRegression() {
        // Does nothing.
    }


    /**
     * {@inheritDoc}
     *
     * @param features Features to use during training of the model.
     * @param targets Targets to use during training of the model.
     */
    @Override
    public MultipleLinearRegression fit(double[][] features, double[] targets) {

        Matrix y = new Vector(targets);
        Matrix V = new Matrix(features);
        Matrix ones = Matrix.ones(features.length, 1);
        V = ones.augment(V);

        Matrix VT = V.T();

        // Form the normal equations
        Matrix A = VT.mult(V);
        Vector b = VT.mult(y).toVector();

        if(A.isSingular()) { // Then we can not explicitly solve Ax=b for a single solution.
            throw new IllegalArgumentException("The data resulted in an equation with a singular matrix. " +
                    "Singular matrices are not supported. Use the MultipleLinearRegressionSGD model instead.");
        }

        w = Solvers.solveQR(A, b); // Compute the model parameters by solving the system via QR decomposition.

        this.coefficients = w.T().getValuesAsDouble()[0];

        isFit = true;
        buildDetails(); // Build the details of the model.

        return this;
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
        if(!isFit) {
            throw new IllegalStateException("Model must be fit before predictions can be made.");
        }

        Matrix X = Matrix.ones(features.length, 1).augment(new Matrix(features));

        return X.mult(w).T().getValuesAsDouble()[0];
    }


    /**
     * {@inheritDoc}
     */
    public Matrix predict(Matrix X, Matrix w) {
        return X.mult(w);
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix getParams() {
        return this.w;
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
        if(!filePath.endsWith(".mdl")) {
            throw new IllegalArgumentException("Incorrect file type. File does not end with \".mdl\".");
        }

        blockList = new Block[2];

        // Construct the blocks for the model file.
        blockList[0] = new Block(LinearModelTags.MODEL_TYPE.toString(), this.MODEL_TYPE);
        blockList[1] = new Block(LinearModelTags.PARAMETERS.toString(), ArrayUtils.asString(this.coefficients));

        FileManager.stringToFile(Block.buildFileContent(blockList), filePath);
    }


    // Construct details of model
    protected void buildDetails() {
        details = new StringBuilder(
                "Model Details\n" +
                        "----------------------------\n" +
                        "Model Type: " + this.MODEL_TYPE+ "\n" +
                        "Is Trained: " + (isFit ? "Yes" : "No") + "\n"
        );

        if(isFit && coefficients!=null) {
            details.append("Coefficients: ");
            details.append(ArrayUtils.asString(coefficients));
            details.append("\nHyperplane: y = ").append(coefficients[0]).append(" + ");

            for(int i=1; i<coefficients.length; i++) {
                details.append(coefficients[i]).append("x_").append(i);

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
    public String inspect() {
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
