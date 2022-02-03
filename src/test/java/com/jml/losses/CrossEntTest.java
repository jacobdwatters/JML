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


    @Test
    void crossEnt3Test() {
        double[][] y = {{0, 0, 0, 1},
                {0, 1, 0, 0},
                {0, 1, 0, 0},
                {1, 0, 0, 0},
                {0, 0, 1, 0}};
        double[][] y_pred = {{0.01, 0.04, 0, 0.95},
                {0.12, 0.8, 0.01, 0.07},
                {0.6, 0.3, 0.025, 0.075},
                {0.452, 0.21, 0.1, 0.238},
                {0, 0, 0.8, 0.2}};

        expectedCE = 0.4991252600983624;

        Matrix Y = new Matrix(y);
        Matrix Y_pred = new Matrix(y_pred);

        Matrix loss = LossFunctions.crossEntropy.compute(Y, Y_pred);

        assertEquals(expectedCE, loss.getAsDouble(0, 0));
    }
}
