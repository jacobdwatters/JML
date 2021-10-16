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
 * Model for least squares linear regression of polynomials.<br><br>
 *
 * PolynomialRegression fits a model y = b<sub>0</sub> + b<sub>1</sub>x  + b<sub>2</sub>x<sup>2</sup> + ... +
 * b<sub>n</sub>x<sup>n</sup> to the datasets by minimizing
 * the residuals of the sum of squares between the values in the target dataset and the values predicted
 * by the model. This is solved explicitly.
 */
public class PolynomialRegression extends Model<double[], double[]> {
    String MODEL_TYPE = ModelTypes.POLYNOMIAL_REGRESSION.toString();

    protected boolean isFit = false;

    protected int degree; // Defaults to simple linear regression.
    protected double[] coefficients;

    // Weights of the model
    protected Matrix w;

    // Details of model in human-readable format.
    private StringBuilder details = new StringBuilder(
            "Model Details\n" +
                    "----------------------------\n" +
                    "Model Type: " + this.MODEL_TYPE+ "\n" +
                    "Is Trained: No\n"
    );


    /**
     * Creates a default polynomial regression model. The default model is a degree one polynomial.
     */
    public PolynomialRegression() {
        this.degree = 1;
    }


    /**
     * Creates a polynomial regression model with specified degree.
     *
     * @param degree Degree of polynomial to fit data.
     */
    public PolynomialRegression(int degree) {
        if(degree < 1) {
            throw new IllegalArgumentException("Degree must be greater than or equal to 1 but got " + degree);
        }

        this.degree = degree;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public PolynomialRegression fit(double[] features, double[] targets) {


        Vector x = new Vector(features);
        Matrix y = (new Vector(targets)).toMatrix();
        Matrix V = Matrix.van(x, degree+1);
        Matrix VT = V.T();

        Matrix A = VT.mult(V);
        Vector b = VT.mult(y).toVector();
        w = Solvers.solve(A, b);
        coefficients = w.T().getValuesAsDouble()[0];

        isFit = true;
        buildDetails(); // Build the details of the model.

        return this;
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
        if(!isFit) {
            throw new IllegalStateException("Model must be fit before predictions can be made.");
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
     * {@inheritDoc}
     */
    @Override
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
            blockList = new Block[2];
            linOrPolyType = ModelTypes.LINEAR_REGRESSION.toString();
        } else {
            blockList = new Block[3];
            linOrPolyType = this.MODEL_TYPE;
            blockList[2] = new Block(LinearModelTags.DEGREE.toString(), Integer.toString(degree));
        }

        // Construct the blocks for the model file.
        blockList[0] = new Block(LinearModelTags.MODEL_TYPE.toString(), linOrPolyType);
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

        details.append("Polynomial Degree: " + degree + "\n");

        if(isFit && w!=null) {
            coefficients = w.T().getValuesAsDouble()[0];
            details.append("Coefficients (low->high): ");
            details.append(ArrayUtils.asString(coefficients));
            details.append("\nPolynomial: y = " + coefficients[0] + " + " + coefficients[1] + "x + ");

            for(int i=2; i<coefficients.length; i++) {
                details.append(coefficients[i] + "x^" + i);

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

    public static void main(String[] args) {
        double[] x = {1, 4, 5, 6, 7};
        double[] y = {3, 4, 5, 6, 10};

        Model model = new LinearRegression();
        model.fit(x, y);

        System.out.println(model.getDetails());
        System.out.println(ArrayUtils.asString(model.getParams().T().getValuesAsDouble()[0]));

        model.saveModel("temp.mdl");

        Model nmdl = Model.load("temp.mdl");
//        System.out.println(nmdl.getDetails());
    }
}
