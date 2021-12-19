package com.jml.losses;

import com.jml.core.Stats;
import linalg.Matrix;
import linalg.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class binCrossEntTest {

    double expectedBCE;

    @Test
    void binCE1Test() {
        double[] exp = {1, 0, 0, 1, 1, 0};
        double[] pred = {0.56, 0.12, 0.98, 0.331, 0.234, 0.02};
        expectedBCE = 1.1996581077896666;

        double bce = LossFunctions.binCrossEntropy.compute(new Vector(exp), new Vector(pred)).getAsDouble(0, 0);

        assertEquals(expectedBCE, bce);
    }

    @Test
    void binCE2Test() {
        double[] exp = {1, 0, 0, 1, 1, 0, 1, 0};
        double[] pred = {0.61077476, 0.04592894, 0.03497691, 0.09895611, 0.39080419, 0.40498168,
                0.03639751, 0.07097484};
        expectedBCE = 0.96678904;

        double bce = LossFunctions.binCrossEntropy.compute(new Vector(exp), new Vector(pred)).getAsDouble(0, 0);

        assertEquals(expectedBCE, Stats.round(bce, 8));
    }

    @Test
    void binCE3Test() {
        double[] exp = {1, 0, 0, 1, 1, 0, 1, 0};
        double[] pred = {0, 1, 0, 1, 1, 0, 1, 0};
        expectedBCE = 8.634794;

        double bce = LossFunctions.binCrossEntropy.compute(new Vector(exp), new Vector(pred)).getAsDouble(0, 0);

        assertEquals(expectedBCE, Stats.round(bce, 6));
    }

    @Test
    void multiClassTest() {
        double[][] exp = {  {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}};
        double[][] pred = { {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0}};

        assertThrows(IllegalArgumentException.class, () -> LossFunctions.binCrossEntropy.compute(new Matrix(exp), new Matrix(pred)));
    }
}
