package com.jml.linear_models;

import com.jml.core.Model;
import org.junit.jupiter.api.BeforeEach;

class LinearRegressionSGDTest {
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


    void defaultConstructorTestCase() {
        model = new LinearRegressionSGD();
        // TODO:
    }
}
