package com.jml.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class StatsTest {
    double[] dataset1, dataset2, dataset3, dataset4;


    @BeforeEach
        // Runs before each test
    void setUp() {
        dataset1 = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        dataset2 = new double[]{1, 2};
        dataset3 = new double[]{-12};
        dataset4 = new double[]{0, 0, 0, 0};
    }

    @Test // Defines a test method
    @DisplayName("Test for median value") // define the name of the test which is displayed to the user
    void medianTestCase() {
        double med1 = 2.98;
        double med2 = 1.5;
        double med3 = -12;
        double med4 = 0;

        assertEquals(med1, Stats.median(dataset1));
        assertEquals(med2, Stats.median(dataset2));
        assertEquals(med3, Stats.median(dataset3));
        assertEquals(med4, Stats.median(dataset4));
    }

    @Test // Defines a test method
    @DisplayName("Test for sst=0 in determination") // define the name of the test which is displayed to the user
    void determinationExceptionTestCase() {
        assertThrows(Exception.class, () -> Stats.determination(dataset4, dataset1));
    }
}
