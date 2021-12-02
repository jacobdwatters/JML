package com.jml.core;

import com.jml.losses.Function;
import com.jml.losses.LossFunctions;

import linalg.Matrix;
import linalg.Vector;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LossFunctionTest {

    Function mse = LossFunctions.mse;
    Function binCrossEntropy = LossFunctions.binCrossEntropy;
    Function crossEntropy = LossFunctions.crossEntropy;
    Function sse = LossFunctions.sse;
    Matrix x, y;
    double loss, expLoss;

    @Test
    void mseTest() {
        x = new Vector(new double[]{1, 5, 6, 2, 7});
        y = new Vector(new double[]{2, 4, 1, 7, 2});
        expLoss = 15.4;

        loss = mse.compute(x, y).getAsDouble(0, 0);

        assertEquals(expLoss, loss);
    }

    @Test
    void sseTest() {
        x = new Vector(new double[]{1, 5, 6, 2, 7});
        y = new Vector(new double[]{2, 4, 1, 7, 2});
        expLoss = 77;

        loss = sse.compute(x, y).getAsDouble(0, 0);

        assertEquals(expLoss, loss);
    }

    @Test
    void bCEntTest() {
        x = new Vector(new double[]{0.2, 0.4, 0.5, 0.98, 0.45});
        y = new Vector(new double[]{0, 1, 0, 1, 1});
        expLoss = 0.530258373457;

        loss = binCrossEntropy.compute(y, x).getAsDouble(0, 0);

        assertEquals(expLoss, Stats.round(loss, 12));
    }
}
