package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.neural_network.activations.Activations;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.RepeatedTest;

import static org.junit.jupiter.api.Assertions.*;

class PerceptronTest {
    Perceptron perceptron;

    final double[][] X1 = {{0.5, 1},
            {0, 0.5},
            {0, 1},
            {0.5, 0},
            {1, 0.5},
            {1, 0}};
    final double[][] y1 = {{1},
            {1},
            {1},
            {0},
            {0},
            {0}};

    @Test
    void constructorTest() {
        perceptron = new Perceptron();

        assertEquals(0.01, perceptron.learningRate);
        assertEquals(10, perceptron.epochs);
        assertEquals(1, perceptron.batchSize);
        assertEquals(1e-5, perceptron.threshold);
        assertEquals(Activations.sigmoid, perceptron.activation);
    }


    @Test
    void constructorOneTest() {
        perceptron = new Perceptron(0.05, 105, 2, 1e-3);

        assertEquals(0.05, perceptron.learningRate);
        assertEquals(105, perceptron.epochs);
        assertEquals(2, perceptron.batchSize);
        assertEquals(1e-3, perceptron.threshold);
        assertEquals(Activations.sigmoid, perceptron.activation);
    }


    @Test
    void constructorTwoTest() {
        perceptron = new Perceptron(0.05, 105, 2, 1e-3, Activations.relu);

        assertEquals(0.05, perceptron.learningRate);
        assertEquals(105, perceptron.epochs);
        assertEquals(2, perceptron.batchSize);
        assertEquals(1e-3, perceptron.threshold);
        assertEquals(Activations.relu, perceptron.activation);
    }


    @RepeatedTest(10)
    void fitTest() {
        String filePath = "./src/test/java/com/jml/linear_models/test_model_files/testPerceptron.mdl";
        perceptron = new Perceptron(0.1, 5000, 6, 1e-5);

        assertEquals(0.1, perceptron.learningRate);
        assertEquals(5000, perceptron.epochs);
        assertEquals(6, perceptron.batchSize);
        assertEquals(1e-5, perceptron.threshold);
        assertEquals(Activations.sigmoid, perceptron.activation);

        perceptron.fit(X1, y1);
        perceptron.saveModel(filePath);
        double[][] predictions = perceptron.predict(X1);

        assertArrayEquals(y1, ArrayUtils.round(predictions, 0));

        Model<double[][], double[][]> loadedPercep = Model.load(filePath);
        double[][] loadedPredictions = loadedPercep.predict(X1);

        assertArrayEquals(predictions, loadedPredictions);
    }
}
