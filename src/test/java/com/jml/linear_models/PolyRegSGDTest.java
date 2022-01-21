package com.jml.linear_models;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PolyRegSGDTest {

    PolynomialRegressionSGD model;
    double[] features;
    double[] targets;


    @BeforeEach
        // Runs before each test
    void setUp() {
        features = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        targets = new double[]{44.6, 57.8, 49.9, 61.3, 49.6, 61.8, 49.0, 44.7, 59.2, 53.9, 46.5, 54.7, 50.3, 51.2, 45.7};
    }

    @Test
    void defaultConstructorTestCase() {
        double learningRate = 0.002;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        model = new PolynomialRegressionSGD();

        assertEquals(1, model.degree);
        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
    }


    @Test
    void defaultConstructorZeroTestCase() {
        double learningRate = 0.002;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        model = new PolynomialRegressionSGD(3);

        assertEquals(3, model.degree);
        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
    }


    @Test
    void ConstructorOneTestCase() {
        int degree = 3;
        double learningRate = 0.051;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        model = new PolynomialRegressionSGD(degree, learningRate);

        assertEquals(degree, model.degree);
        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
    }


    @Test
    void ConstructorTwoTestCase() {
        int degree = 5;
        double learningRate = 0.6;
        double threshold = 0.5e-5;
        int maxIterations = 2048;

        model = new PolynomialRegressionSGD(degree,learningRate, maxIterations);

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
    }

    @Test
    void ConstructorThreeTestCase() {
        int degree = 3;
        double learningRate = 0.012;
        double threshold = 0.0005;
        int maxIterations = 1024;

        model = new PolynomialRegressionSGD(degree,learningRate, maxIterations, threshold);

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
    }


    @Test
    void dataFitTest() {
        int degree = 2;
        double learningRate = 0.051;
        double threshold = 0.5e-5;
        int maxIterations = 200;

        model = new PolynomialRegressionSGD(degree, learningRate, maxIterations, threshold);
        assertTrue(!model.isFit);

        model.fit(features, targets);

        double[] lossHist = model.getLossHist();

        assertTrue(model.isFit);
    }


    @Test
    void errorTest() {
        double learningRate = -1;
        int maxIterations = -5;
        double threshold = -9;

        assertThrows(Exception.class, () -> new PolynomialRegressionSGD(1, learningRate, 1, 1));
        assertThrows(Exception.class, () -> new PolynomialRegressionSGD(1, 1, maxIterations, 1));
        assertThrows(Exception.class, () -> new PolynomialRegressionSGD(1, 1, 1, threshold));
    }

    // Average difference of neighboring elements.
    protected static double aveSlope(double[] arr) {
        return (arr[0]-arr[arr.length-1])/2;
    }
}
