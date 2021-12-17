package com.jml.losses;

import com.jml.core.Stats;
import linalg.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CrossEntTest {
    double expectedCE;

    @Test
    void crossEnt1Test() {
        double[][] exp = {  {1, 0, 0},
                            {0, 1, 0},
                            {0, 0, 1}};
        double[][] pred = { {0.85, 0.12, 0.03},
                            {0.21, 0.14, 0.65},
                            {0.56, 0.22, 0.22}};
        expectedCE = 1.2142532;

        double ce = LossFunctions.crossEntropy.compute(new Matrix(exp), new Matrix(pred)).getAsDouble(0, 0);

        assertEquals(expectedCE, Stats.round(ce, 7));
    }


    @Test
    void crossEnt2Test() {
        double[][] exp = {  {1, 0, 0},
                            {0, 1, 0},
                            {0, 0, 1}};
        double[][] pred = { {1, 0, 0},
                            {1, 0, 0},
                            {0, 1, 0}};
        expectedCE = 23.025851;

        double ce = LossFunctions.crossEntropy.compute(new Matrix(exp), new Matrix(pred)).getAsDouble(0, 0);

        assertEquals(expectedCE, Stats.round(ce, 6));
    }
}
