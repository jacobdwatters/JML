package com.jml.classifiers.naive_bayes;

import com.jml.core.Model;
import com.jml.preprocessing.DataSplitter;

import com.jml.util.ArrayUtils;
import linalg.Matrix;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class GaussianNaiveBayes extends Model<double[][], double[]> {

    private Map<Integer, List<double[]>> data;
    private Map<Integer, double[]> meanByCol; // Mean for each column of the features for each class.
    private Map<Integer, double[]> stddevByCol; // Standard deviation for each column of the features for each class.


    @Override
    public GaussianNaiveBayes fit(double[][] features, double[] targets) {
        data = DataSplitter.splitByClass(features, ArrayUtils.toInt(targets));
        summerize();

        return this;
    }


    // Computes the mean and standard deviation for each feature for each class.
    private void summerize() {
        Matrix classData; // Data for a single class

        for(Integer key : data.keySet()) {
            // TODO: compute mean/std for each feature column in this class.
            classData = new Matrix(ArrayUtils.toDouble2D(data.get(key).toArray())).T();
        }

    }


    @Override
    public double[] predict(double[][] features) {
        // TODO: Auto-generated method stub.
        return new double[0];
    }


    @Override
    public Matrix predict(Matrix X, Matrix w) {
        // TODO: Auto-generated method stub.
        return null;
    }


    @Override
    public Matrix getParams() {
        // TODO: Auto-generated method stub.
        return null;
    }


    @Override
    public void saveModel(String filePath) {
        // TODO: Auto-generated method stub.
    }


    @Override
    public String inspect() {
        // TODO: Auto-generated method stub.
        return null;
    }


    @Override
    public String toString() {
        // TODO: Auto-generated method stub.
        return null;
    }


    public static void main(String[] args) {
        double[][] x = {{1, 2, 3},
                {3, 2, 1},
                {2, 3, 1},
                {-1, -2, -3},
                {8, 9, 10},
                {12, -2, -1},
                {0.23, 3.5, 9.8}};
        double y[] = {0, 0, 1, 0, 2, 1, 2};

        GaussianNaiveBayes nb = new GaussianNaiveBayes();
        nb.fit(x, y);
    }
}
