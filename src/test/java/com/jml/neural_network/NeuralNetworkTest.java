package com.jml.neural_network;

import com.jml.core.Model;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.RepeatedTest;
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
        assertArrayEquals(Y, ArrayUtils.round(pred, 0));

        nn.saveModel(filePath);

        Model<double[][], double[][]> loadedNN = Model.load(filePath);
        double[][] loadedPred = loadedNN.predict(X);

        assertEquals(nn.inspect(), loadedNN.inspect());
        assertArrayEquals(pred, loadedPred);
    }
}
