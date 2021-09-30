package com.jml.preprocessing;

import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NormalizationTest {

    double[] dataOne;
    double[] dataTwo;
    double[][] data2D;
    double[] expectedOne;
    double[] expectedTwo;
    double[][] expected2D;

    @BeforeEach
    void setUp() {
        dataOne = new double[]{1, 2, 3, 4, 5};
        dataTwo = new double[]{0.12, 12.2342, -123.92342, 193.9244};
    }

    @Test
    void l2NormalizeTestCase() {
        expectedOne = new double[]{0.13483997, 0.26967994, 0.40451992, 0.53935989, 0.67419986};

        System.out.println("\n\n" + ArrayUtils.asString(Normalize.l2(dataOne)) + "\n\n");
    }
}
