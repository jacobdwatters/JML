package com.jml.linear_models;

import com.jml.core.Model;
import linalg.Matrix;


/**
 * Fits a one layer perceptron to a set of features.<br><br>
 *
 * Perceptron is a linear classifier that is analogous to logistic regression using stochastic gradient descent.
 */
public class Perceptron extends Model<double[][], double[][]> {


    /**
     * {@inheritDoc}
     */
    @Override
    public Perceptron fit(double[][] features, double[][] targets) {
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public double[][] predict(double[][] features) {
        // TODO: Auto-generated method stub
        return null;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Auto-generated method stub.
        return null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix getParams() {
        return null;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void saveModel(String filePath) {
        // TODO: Auto-generated method stub
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public String getDetails() {
        return this.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        // TODO: Auto-generated method stub.
        return "";
    }
}
