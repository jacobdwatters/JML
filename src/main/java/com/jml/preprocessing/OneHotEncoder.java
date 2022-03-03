package com.jml.preprocessing;

import java.util.Map;


/**
 * An object that allows for the categorical encoding of samples as one-hot arrays.
 */
public class OneHotEncoder {

    /* Flag for handling unknown samples. If false, a sample that was not seen during the fit() call
        will cause an error. If true, the sample will be "ignored" and all zeros will be returned.
     */
    private boolean ignoreUnknown;
    Map<String, double[]> encodings;


    /**
     * Creates a OneHotEncoder object.
     */
    public OneHotEncoder() {
        this.ignoreUnknown = false;
    }


    /**
     * Creates a OneHotEncoder object.
     * @param ignoreUnknown Flag for handling unknown samples. If false, a sample passed to encode() that was not seen
     *                      during the fit() call will cause an error. If true, the sample will be "ignored" and all
     *                      zeros will be returned.
     */
    public OneHotEncoder(boolean ignoreUnknown) {
        this.ignoreUnknown = ignoreUnknown;
    }



    public double[][] fit(double[] features) {
        // TODO:
        return null;
    }

    public double[][] fit(double[][] features) {
        // TODO:
        return null;
    }


    public double[][] fit(String[] features) {
        // TODO:
        return null;
    }

    public double[][] fit(String[][] features) {
        // TODO:
        return null;
    }






    public double[][] encode(double[] features) {
        // TODO:
        return null;
    }

    public double[][] encode(double[][] features) {
        // TODO:
        return null;
    }


    public double[][] encode(String[] features) {
        // TODO:
        return null;
    }

    public double[][] encode(String[][] features) {
        // TODO:
        return null;
    }



    public double[][] invEncode(double[] features) {
        // TODO:
        return null;
    }

    public double[][] invEncode(double[][] features) {
        // TODO:
        return null;
    }


    public double[][] invEncode(String[] features) {
        // TODO:
        return null;
    }

    public double[][] invEncode(String[][] features) {
        // TODO:
        return null;
    }
}
