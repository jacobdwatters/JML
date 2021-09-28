package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class LinearRegressionDatasetOneTest {
    Model<double[], double[]> model;
    double[] features;
    double[] targets;
    double[] tests;


    @BeforeEach
        // Runs before each test
    void setUp() {
        model = new LinearRegression();
        features = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        targets = new double[]{44.6, 57.8, 49.9, 61.3, 49.6, 61.8, 49.0, 44.7, 59.2, 53.9, 46.5, 54.7, 50.3, 51.2, 45.7};
        tests = new double[]{-1.2, 0, 1, 2, 3.14, 4};
    }


    @Test // Defines a test method
    @DisplayName("Testing for Linear Regression model.") // define the name of the test which is displayed to the user
    void DatasetOneTestCase() {
        double[] expected = {45.65396403, 2.34259675};
        double[] testingExpected = {42.8428479, 45.6539640, 47.9965608, 50.3391575, 53.0097178, 55.0243510};

        model.compile();
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }


    @Test // Defines a test method
    @DisplayName("Testing for Linear Regression model.") // define the name of the test which is displayed to the user
    void DatasetOneSecondTestCase() {
        double[] expected = {45.65396403, 2.34259675};
        double[] testingExpected = {42.8428479, 45.6539640, 47.9965608, 50.3391575, 53.0097178, 55.0243510};

        Map<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 3.0);

        model.compile(arguments); // Attempt to add a degree arguments
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }
}
