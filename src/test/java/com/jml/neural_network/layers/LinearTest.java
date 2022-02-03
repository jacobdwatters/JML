package com.jml.neural_network.layers;

import com.jml.neural_network.activations.ActivationFunction;
import com.jml.neural_network.activations.Tanh;
import com.jml.neural_network.layers.initilizers.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LinearTest {
    static Linear layer;
    final static int inDim = 4;
    final static int outDim = 9;
    final static Initializer wInit = new HeUniform();
    final static Initializer bInit = new Constant(5);


    @Test
    void constructor1Test() {
        layer = new Linear(inDim, outDim);

        assertEquals(layer.getInDim(), inDim);
        assertEquals(layer.getOutDim(), outDim);
        assertTrue(layer.weightInitializer instanceof GlorotNormal);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor2Test() {
        layer = new Linear(inDim, outDim, wInit);

        assertEquals(layer.getInDim(), inDim);
        assertEquals(layer.getOutDim(), outDim);
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor3Test() {
        layer = new Linear(inDim, outDim, wInit, bInit);

        assertEquals(layer.getInDim(), inDim);
        assertEquals(layer.getOutDim(), outDim);
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Constant);
    }


    @Test
    void constructor4Test() {
        layer = new Linear(outDim);

        assertEquals(layer.getInDim(), -1);
        assertEquals(layer.getOutDim(), outDim);
        assertTrue(layer.weightInitializer instanceof GlorotNormal);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor5Test() {
        layer = new Linear(outDim, wInit);

        assertEquals(layer.getInDim(), -1);
        assertEquals(layer.getOutDim(), outDim);
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor6Test() {
        layer = new Linear(outDim, wInit, bInit);

        assertEquals(layer.getInDim(), -1);
        assertEquals(layer.getOutDim(), outDim);
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Constant);
    }
}
