package com.jml.neural_network.layers;

import com.jml.neural_network.activations.ActivationFunction;
import com.jml.neural_network.activations.Tanh;

import com.jml.neural_network.layers.initilizers.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DenseTest {
    static Dense layer;
    final static int inDim = 4;
    final static int outDim = 9;
    final static ActivationFunction activation = new Tanh();
    final static Initializer wInit = new HeUniform();
    final static Initializer bInit = new Constant(5);


    @Test
    void constructor1Test() {
        layer = new Dense(inDim, outDim, activation);

        assertEquals(layer.getInDim(), inDim);
        assertEquals(layer.getOutDim(), outDim);
        assertEquals(layer.activation.getName(), activation.getName());
        assertTrue(layer.weightInitializer instanceof GlorotNormal);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor2Test() {
        layer = new Dense(inDim, outDim, activation, wInit);

        assertEquals(layer.getInDim(), inDim);
        assertEquals(layer.getOutDim(), outDim);
        assertEquals(layer.activation.getName(), activation.getName());
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor3Test() {
        layer = new Dense(inDim, outDim, activation, wInit, bInit);

        assertEquals(layer.getInDim(), inDim);
        assertEquals(layer.getOutDim(), outDim);
        assertEquals(layer.activation.getName(), activation.getName());
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Constant);
    }


    @Test
    void constructor4Test() {
        layer = new Dense(outDim, activation);

        assertEquals(layer.getInDim(), -1);
        assertEquals(layer.getOutDim(), outDim);
        assertEquals(layer.activation.getName(), activation.getName());
        assertTrue(layer.weightInitializer instanceof GlorotNormal);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor5Test() {
        layer = new Dense(outDim, activation, wInit);

        assertEquals(layer.getInDim(), -1);
        assertEquals(layer.getOutDim(), outDim);
        assertEquals(layer.activation.getName(), activation.getName());
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Zeros);
    }


    @Test
    void constructor6Test() {
        layer = new Dense(outDim, activation, wInit, bInit);

        assertEquals(layer.getInDim(), -1);
        assertEquals(layer.getOutDim(), outDim);
        assertEquals(layer.activation.getName(), activation.getName());
        assertTrue(layer.weightInitializer instanceof HeUniform);
        assertTrue(layer.biasInitializer instanceof Constant);
    }
}
