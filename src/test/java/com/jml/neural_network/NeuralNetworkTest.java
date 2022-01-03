package com.jml.neural_network;

import com.jml.core.Model;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Dropout;
import com.jml.optimizers.*;
import org.junit.jupiter.api.Test;


import static org.junit.jupiter.api.Assertions.*;


class NeuralNetworkTest {
    NeuralNetwork nn;
    double[][] X = {{0, 0},
            {0, 1},
            {1, 0},
            {1, 1}};
    double[][] Y = {{0}, {1}, {1}, {0}};


    @Test
    void constructorTest() {
        nn = new NeuralNetwork();
        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, 5, Activations.relu));

        assertEquals(0.01, nn.learningRate);
        assertEquals(10, nn.epochs);
        assertEquals(1, nn.batchSize);
        assertEquals(1e-5, nn.threshold);
        assertEquals(2, nn.layers.size());
    }


    @Test
    void constructor2Test() {
        nn = new NeuralNetwork(0.5, 100);
        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, 5, Activations.relu));

        assertEquals(0.5, nn.learningRate);
        assertEquals(100, nn.epochs);
        assertEquals(1, nn.batchSize);
        assertEquals(1e-5, nn.threshold);
        assertEquals(2, nn.layers.size());
    }


    @Test
    void constructor3Test() {
        nn = new NeuralNetwork(0.5, 100, 12);
        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, 5, Activations.relu));

        assertEquals(0.5, nn.learningRate);
        assertEquals(100, nn.epochs);
        assertEquals(12, nn.batchSize);
        assertEquals(1e-5, nn.threshold);
        assertEquals(2, nn.layers.size());
    }


    @Test
    void constructor4Test() {
        Optimizer optim = new GradientDescent(0.01);
        nn = new NeuralNetwork(optim, 100);
        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, 5, Activations.relu));

        assertEquals(0.01, nn.learningRate);
        assertEquals(100, nn.epochs);
        assertEquals(1, nn.batchSize);
        assertEquals(1e-5, nn.threshold);
        assertEquals(2, nn.layers.size());
    }


    @Test
    void fitTest() {
        String filePath = "./src/test/java/com/jml/neural_network/test_files/testNN.mdl";
        nn = new NeuralNetwork(0.1, 1000, 2, 0.5e-5);
        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, Activations.relu));
        nn.add(new Dense(1, Activations.sigmoid));

        nn.fit(X, Y);
        double[][] pred = nn.predict(X);

        assertEquals(0.1, nn.learningRate);
        assertEquals(1000, nn.epochs);
        assertEquals(2, nn.batchSize);
        assertEquals(0.5e-5, nn.threshold);
        assertEquals(3, nn.layers.size());

        nn.saveModel(filePath);

        Model<double[][], double[][]> loadedNN = Model.load(filePath);
        double[][] loadedPred = loadedNN.predict(X);

        assertEquals(nn.inspect(), loadedNN.toString());
        assertArrayEquals(pred, loadedPred);
    }


    @Test
    void fit2Test() {
        String filePath = "./src/test/java/com/jml/neural_network/test_files/testNN1.mdl";
        nn = new NeuralNetwork(0.001, 5, 2, 0.5e-5);
        nn.add(new Dropout(2, 0.1));
        nn.add(new Dense(2, 10, Activations.tanh));
        nn.add(new Dense(10, Activations.linear));
        nn.add(new Dropout(0.2));
        nn.add(new Dense(10, Activations.relu));
        nn.add(new Dense(1, Activations.sigmoid));

        nn.fit(X, Y);
        double[][] pred = nn.predict(X);

        assertEquals(0.001, nn.learningRate);
        assertEquals(5, nn.epochs);
        assertEquals(2, nn.batchSize);
        assertEquals(0.5e-5, nn.threshold);
        assertEquals(6, nn.layers.size());

        nn.saveModel(filePath);

        Model<double[][], double[][]> loadedNN = Model.load(filePath);
        double[][] loadedPred = loadedNN.predict(X);

        assertArrayEquals(pred, loadedPred);
    }


    @Test
    void momentumTest() {
        String filePath = "./src/test/java/com/jml/neural_network/test_files/testNN2.mdl";
        Optimizer optim = new Momentum(0.01);
        nn = new NeuralNetwork(optim, 1000, 2, 0.5e-5);
        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, Activations.relu));
        nn.add(new Dense(1, Activations.sigmoid));

        nn.fit(X, Y);
        double[][] pred = nn.predict(X);

        assertEquals(0.01, nn.learningRate);
        assertEquals(1000, nn.epochs);
        assertEquals(2, nn.batchSize);
        assertEquals(0.5e-5, nn.threshold);
        assertEquals(3, nn.layers.size());
        assertEquals(optim, nn.optim);

        nn.saveModel(filePath);

        Model<double[][], double[][]> loadedNN = Model.load(filePath);
        double[][] loadedPred = loadedNN.predict(X);

        assertArrayEquals(pred, loadedPred);
    }


    @Test
    void schedTest() {
        String filePath = "./src/test/java/com/jml/neural_network/test_files/testNN3.mdl";
        Optimizer optim = new GradientDescent(0.01);
        Scheduler schedule = new StepLearningRate(optim, 0.5, 100);

        nn = new NeuralNetwork(optim, 1000, 10);
        nn.setScheduler(schedule);

        nn.add(new Dense(2, 10, Activations.sigmoid));
        nn.add(new Dense(10, Activations.relu));
        nn.add(new Dense(1, Activations.sigmoid));

        nn.fit(X, Y);
        double[][] pred = nn.predict(X);

        assertEquals(0.01, nn.learningRate);
        assertEquals(1000, nn.epochs);
        assertEquals(10, nn.batchSize);
        assertEquals(1e-5, nn.threshold);
        assertEquals(3, nn.layers.size());
        assertEquals(optim, nn.optim);

        nn.saveModel(filePath);

        Model<double[][], double[][]> loadedNN = Model.load(filePath);
        double[][] loadedPred = loadedNN.predict(X);

        assertArrayEquals(pred, loadedPred);
    }
}
