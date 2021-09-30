package com.jml.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class NormalizationExceptionTest {
    double[] dataOne;
    double[] dataTwo;

    @BeforeEach
    void setUp() {
        dataOne = new double[]{1, 1, 1, 1, 1};
    }


    @Test
    void maxMinEqualException() {
        assertThrows(Exception.class, () -> Normalize.minMaxScale(dataOne));
        assertThrows(Exception.class, () -> Normalize.meanNormalize(dataOne));
    }

    @Test
    void maxMinException() {
        assertThrows(Exception.class, () -> Normalize.minMaxScale(dataOne, 6, 2));
    }
}
