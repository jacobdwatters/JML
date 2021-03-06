package com.jml.linear_models;

import com.jml.core.Model;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.assertThrows;

class PolynomialRegressionExceptionTest {

    Model<double[], double[]> model;
    double[] features;
    double[] targets;
    double[] tests;


    @BeforeEach
        // Runs before each test
    void setUp() {
        features = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        targets = new double[]{44.6, 57.8, 49.9, 61.3, 49.6, 61.8, 49.0, 44.7, 59.2, 53.9, 46.5, 54.7, 50.3, 51.2, 45.7};
        tests = new double[]{-1.2, 0, 1, 2, 3.14, 4};
    }


    @Test // Defines a test method
    @DisplayName("Test for attempted fitting/prediction but model is not compiled yet.") // define the name of the test which is displayed to the user
    void notCompiledTestCase() {
        assertThrows(Exception.class, () -> model.fit(features, targets));
        assertThrows(Exception.class, () -> model.predict(tests));
    }


    @Test // Defines a test method
    @DisplayName("Test for attempted prediction but model is not fit.") // define the name of the test which is displayed to the user
    void notFitTestCase() {
        double[] expected = {45.96048699, 1.95748083, 0.06892953};
        double[] testingExpected = {43.7107685, 45.9604870, 47.9868974, 50.1511668, 52.7865944, 54.8932828};

        HashMap<String, Double> arguments = new HashMap<>();
        model = new PolynomialRegression(2);

        assertThrows(Exception.class, () -> model.predict(tests));
    }


    @Test // Defines a test method
    @DisplayName("Test for exception when degree is invalid") // define the name of the test which is displayed to the user
    void invalidDegreeCase() {
        assertThrows(Exception.class, () -> new PolynomialRegression(-1));
        assertThrows(Exception.class, () -> new PolynomialRegression(0));
    }
}
