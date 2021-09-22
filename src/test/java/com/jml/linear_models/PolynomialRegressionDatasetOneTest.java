
package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import java.util.HashMap;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;


// Contains test cases for polynomial regression of various degrees with no normalization.
class PolynomialRegressionDatasetOneTest {
    Model<double[], double[]> model;
    double[] features;
    double[] targets;
    double[] tests;


    @BeforeEach // Runs before each test
    void setUp() {
        model = new PolynomialRegression();
        features = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        targets = new double[]{44.6, 57.8, 49.9, 61.3, 49.6, 61.8, 49.0, 44.7, 59.2, 53.9, 46.5, 54.7, 50.3, 51.2, 45.7};
        tests = new double[]{-1.2, 0, 1, 2, 3.14, 4};
    }


    @Test // Defines a test method
    @DisplayName("Testing for degree 1") // define the name of the test which is displayed to the user
    void degreeOneTestCase() {
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
    @DisplayName("Testing for degree 2") // define the name of the test which is displayed to the user
    void degreeTwoTestCase() {
        double[] expected = {45.96048699, 1.95748083, 0.06892953};
        double[] testingExpected = {43.7107685, 45.9604870, 47.9868974, 50.1511668, 52.7865944, 54.8932828};

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 2.0);

        model.compile(arguments);
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }


    @Test // Defines a test method
    @DisplayName("Testing for degree 3") // define the name of the test which is displayed to the user
    void degreeThreeTestCase() {
        double[] expected = {45.95457914, 0.79220291, 0.67392271, -0.06934747};
        double[] testingExpected = {46.0942168, 45.9545791, 47.3513573, 49.6798961, 52.9397664, 55.4679162};

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 3.0);

        model.compile(arguments);
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }


    @Test // Defines a test method
    @DisplayName("Testing for degree 4") // define the name of the test which is displayed to the user
    void degreeFourTestCase() {
        double[] expected = {46.37140013, 2.21354807, -1.01447580, 0.44798964, -0.04608911};
        double[] testingExpected = {41.3846008, 46.3714001, 47.9723729, 49.5870844, 52.7085899, 55.8665043};

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 4.0);

        model.compile(arguments);
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }


    @Test // Defines a test method
    @DisplayName("Testing for degree 5") // define the name of the test which is displayed to the user
    void degreeFiveTestCase() {
        double[] expected = {46.32787494, 2.13168183, -0.81750590, 0.34053349, -0.02399977, -0.00153648};
        double[] testingExpected = {41.9582637, 46.3278749, 47.9570481, 49.6123191, 52.7016377, 55.8513508};

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 5.0);

        model.compile(arguments);
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }


    @Test // Defines a test method
    @DisplayName("Testing for degree 6") // define the name of the test which is displayed to the user
    void degreeSixTestCase() {
        double[] expected = {48.0173315, 2.5733680, -10.6886001, 9.9701278, -3.5275208, 0.5457470, -0.0309329};
        double[] testingExpected = {3.5442995, 48.0173315, 46.8595204, 49.2145514, 53.4008314, 54.4796373};

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 6.0);

        model.compile(arguments);
        double[][] c  = model.fit(features, targets);
        double[] values = ArrayUtils.round(c[0], 7);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        assertArrayEquals(expected, values);
        assertArrayEquals(testingExpected, predictions);
    }
}
