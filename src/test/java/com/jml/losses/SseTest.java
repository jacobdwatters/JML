package com.jml.losses;

import linalg.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SseTest {
    double expectedMSE;

    @Test
    void mse1Test() {
        double[] exp = {3, -0.5, 2, 7};
        double[] pred = {2.5, 0.0, 2, 8};
        expectedMSE = 0.375*4;

        double mse = LossFunctions.sse.compute(new Vector(exp), new Vector(pred)).getAsDouble(0, 0);

        assertEquals(expectedMSE, mse);
    }

    @Test
    void mse2est() {
        double[] exp = {0.8603121767329961, 0.43282841782243653, 0.2324082241907981, 0.10318251270693346, 0.77579305379459};
        double[] pred = {0.35002722928014973, 0.39416945398735603, 0.35438058593383137, 0.8160923444047998, 0.0842375327532473};
        expectedMSE = 0.25265039338503453*5;

        double mse = LossFunctions.sse.compute(new Vector(exp), new Vector(pred)).getAsDouble(0, 0);

        assertEquals(expectedMSE, mse);
    }
}
