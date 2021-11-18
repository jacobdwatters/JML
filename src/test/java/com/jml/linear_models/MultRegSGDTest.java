package com.jml.linear_models;

import com.jml.optimizers.Scheduler;
import com.jml.optimizers.StepLearningRate;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MultRegSGDTest {

    MultipleLinearRegressionSGD model;
    double[][] features;
    double[] targets;


    @BeforeEach
        // Runs before each test
    void setUp() {
        features = new double[][]{{1, 2, 6},
                                     {4, 0, 5},
                                     {7, 8, 2},
                                     {-1, -2, -4},
                                     {7, 5, 5, 4}};

        targets = new double[]{44.6, 57.8, 49.9, 61.3, 49.6};
    }


    @Test
    void defaultConstructorTestCase() {
        double learningRate = 0.002;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        model = new MultipleLinearRegressionSGD();

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
        assertNull(model.schedule);
    }


    @Test
    void ConstructorOneTestCase() {
        double learningRate = 0.051;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        model = new MultipleLinearRegressionSGD(0.051);

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
        assertNull(model.schedule);
    }


    @Test
    void ConstructorTwoTestCase() {
        double learningRate = 0.6;
        double threshold = 0.5e-5;
        int maxIterations = 2048;

        model = new MultipleLinearRegressionSGD(0.6, 2048);

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
        assertNull(model.schedule);
    }

    @Test
    void ConstructorThreeTestCase() {
        double learningRate = 0.012;
        double threshold = 0.0005;
        int maxIterations = 1024;

        model = new MultipleLinearRegressionSGD(0.012, 1024, 0.0005);

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
        assertNull(model.schedule);
    }

    @Test
    void ConstructorFourTestCase() {
        double learningRate = 0.012;
        double threshold = 0.0005;
        int maxIterations = 1024;
        Scheduler schedule = new StepLearningRate(0.8);

        model = new MultipleLinearRegressionSGD(0.012, 1024, 0.0005, schedule);

        assertEquals(learningRate, model.learningRate);
        assertEquals(threshold, model.threshold);
        assertEquals(maxIterations, model.maxIterations);
        assertEquals(schedule, model.schedule);
    }

    @Test
    void dataFitTest() {
        double learningRate = 0.0051;
        double threshold = 0.5e-5;
        int maxIterations = 200;

        model = new MultipleLinearRegressionSGD(learningRate, maxIterations, threshold);
        assertTrue(!model.isFit);

        model.fit(features, targets);

        double[] lossHist = model.getLossHist();

        assertTrue(model.isFit);
        assertTrue(aveSlope(lossHist) < 0);
    }


    @Test
    void errorTest() {
        double learningRate = -1;
        int maxIterations = -5;
        double threshold = -9;

        assertThrows(Exception.class, () -> new MultipleLinearRegressionSGD(learningRate, 1, 1));
        assertThrows(Exception.class, () -> new MultipleLinearRegressionSGD(1, maxIterations, 1));
        assertThrows(Exception.class, () -> new MultipleLinearRegressionSGD(1, 1, threshold));
    }


    // Average difference of neighboring elements.
    protected static double aveSlope(double[] arr) {
        return (arr[0]-arr[arr.length-1])/2;
    }
}
