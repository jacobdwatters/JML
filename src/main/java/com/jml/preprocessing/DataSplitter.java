package com.jml.preprocessing;


import com.jml.util.ArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A class that provides a method for splitting a dataset into a training and testing dataset.
 */
public class DataSplitter {
    private DataSplitter(){
        throw new IllegalStateException("Utility class cannot be instantiated.");
    }


    /**
     * Splits a dataset with features and targets randomly into disjoint training and testing datasets.
     * The arrays are first shuffled using the Fisher–Yates algorithm such that both X and y are shuffled the same.
     * Then the arrays are split into a training and testing dataset.
     *
     * @param X Features of the dataset.
     * @param y Targets of the dataset.
     * @param testSize Percent of data to include in the testing dataset. Should be a value between 0 and 1 (inclusive).
     * @return A hashmap containing the split dataset. Use the following keys to obtain the split data:
     * <pre>
     *     key: "xTrain" -> X training data.
     *     key: "yTrain" -> y training data.
     *     key: "xTest" -> X testing data.
     *     key: "yTest" -> y testing data.
     * </pre>
     */
    public static Map<String, double[][]> trainTestSplit(double[][] X, double[][] y, double testSize) {
        if(testSize>1 || testSize<0) {
            throw new IllegalArgumentException("testSize must be a percentage between zero and one (inclusive) but got " +
                    testSize + ".");
        }

        Map<String, double[][]> split = new HashMap<>(4, 1);
        int trainNumber = (int) ((1-testSize)*X.length);
        int testNumber = X.length-trainNumber;

        double[][] trainX = new double[trainNumber][X.length];
        double[][] trainY = new double[trainNumber][y.length];
        double[][] testX = new double[testNumber][X.length];
        double[][] testY = new double[testNumber][y.length];

        double[][][] shuffle = ArrayUtils.shuffle(X, y); // Shuffle the array.

        // Split the dataset based off the computed sizes.
        for(int i=0; i<X.length; i++) {
            if(i<trainNumber) {
                trainX[i] = shuffle[0][i];
                trainY[i] = shuffle[1][i];
            } else {
                testX[i-trainNumber] = shuffle[0][i];
                testY[i-trainNumber] = shuffle[1][i];
            }
        }

        // Insert all subsets into the hashmap.
        split.put("xTrain", trainX);
        split.put("yTrain", trainY);
        split.put("xTest", testX);
        split.put("yTest", testY);

        return split;
    }
}
