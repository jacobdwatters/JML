package com.jml.optimizers;


import com.jml.linear_models.LinearRegression;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class StepLearningRateTest {
    Optimizer optim;
    Scheduler schedule;
    double[] expectedLR;
    double[] actualLR;

    @Test
    void stepTestCase() {
        optim = new StochasticGradientDescent(new LinearRegression(), 1);
        schedule = new StepLearningRate(0.5, 1);
        expectedLR = new double[]{1, 0.5, 0.25, 0.125, 0.0625};
        actualLR = new double[5];
        actualLR[0] = 1;

        for(int i=0; i<4; i++) {
            optim.iterations++;
            schedule.apply(optim);
            actualLR[i+1] = optim.learningRate;
        }

        assertArrayEquals(expectedLR, actualLR);
    }
}
