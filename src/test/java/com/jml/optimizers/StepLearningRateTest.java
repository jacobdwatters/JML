package com.jml.optimizers;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class StepLearningRateTest {
    Optimizer optim;
    Scheduler schedule;
    double[] expectedLR;
    double[] actualLR;

    @Test
    void stepTestCase() {
        optim = new GradientDescent(1);
        schedule = new StepLearningRate(optim,0.5, 1);
        expectedLR = new double[]{1, 0.5, 0.25, 0.125, 0.0625};
        actualLR = new double[5];

        for(int i=0; i<5; i++) {
            schedule.step();
            actualLR[i] = optim.learningRate;
        }

        assertArrayEquals(expectedLR, actualLR);
    }
}
