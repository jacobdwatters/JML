package com.jml.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class StatsTest {
    double[] dataset1, dataset2, dataset3, dataset4;


    @BeforeEach // Runs before each test
    void setUp() {
        dataset1 = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        dataset2 = new double[]{1, 2};
        dataset3 = new double[]{-12};
        dataset4 = new double[]{0, 0, 0, 0};
    }


    @Test
    void roundTestCase() {
        double value = 123123.0933469913853;
        double expected = 123123.093347;

        assertEquals(expected, Stats.round(value, 7));
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


    @Test
    void meanTestCase() {
        double mean1 = 2.7147;
        double mean2 = 1.5;
        double mean3 = -12;
        double mean4 = 0;

        assertEquals(mean1, Stats.round(Stats.mean(dataset1), 4));
        assertEquals(mean2, Stats.mean(dataset2));
        assertEquals(mean3, Stats.mean(dataset3));
        assertEquals(mean4, Stats.mean(dataset4));
    }


    @Test
    void modeTestCase() {
        double mode1 = 1.49;
        double mode2 = 1;
        double mode3 = -12;
        double mode4 = 0;
        double mode5 = 4;

        double[] dataset5 = {1, 43, 4,5, 1, 344, 4, 4, 1, 3, 4, 123, 4};

        assertEquals(mode1, Stats.mode(dataset1));
        assertEquals(mode2, Stats.mode(dataset2));
        assertEquals(mode3, Stats.mode(dataset3));
        assertEquals(mode4, Stats.mode(dataset4));
        assertEquals(mode5, Stats.mode(dataset5));
    }


    @Test
    void varianceTestCase() {
        double variance1 = 3.5901266667;
        double variance2 = 0.5;
        double variance4 = 0;

        assertEquals(variance1, Stats.round(Stats.variance(dataset1), 10));
        assertEquals(variance2, Stats.variance(dataset2));
        assertThrows(Exception.class, () -> Stats.variance(dataset3));
        assertEquals(variance4, Stats.variance(dataset4));
    }

    @Test
    void stdTestCase() {
        double std1 = 1.8305149245195707;
        double std2 = 0.5;
        double std3 = 0;
        double std4 = 0;

        assertEquals(std1, Stats.std(dataset1));
        assertEquals(std2, Stats.std(dataset2));
        assertEquals(std3, Stats.std(dataset3));
        assertEquals(std4, Stats.std(dataset4));
    }


    @Test
    void determinationTestCase() {
        double determination1 = -9.960062955730699;
        double determination2 = -463.0;

        double[] dataset11 = new double[]{6, 2, 6, 1, -12, 4, 6.34, 11, 4, 5, 2, 1, 1, 6, 8};
        double[] dataset22 = new double[]{7, -12};
        double[] dataset33 = new double[]{4};
        double[] dataset44 = new double[]{1, 2, 3, 4};

        assertEquals(determination1, Stats.determination(dataset1, dataset11));
        assertEquals(determination2, Stats.determination(dataset2, dataset22));
        assertThrows(Exception.class, () -> Stats.determination(dataset3, dataset33));
        assertThrows(Exception.class, () ->  Stats.determination(dataset4, dataset44));
    }


    @Test
    void minIndexTestCase() {
        int minIndex1 = 14;
        int minIndex2 = 0;
        int minIndex3 = 0;
        int minIndex4 = 3;

        assertEquals(minIndex1, Stats.minIndex(dataset1));
        assertEquals(minIndex2, Stats.minIndex(dataset2));
        assertEquals(minIndex3, Stats.minIndex(dataset3));
        assertEquals(minIndex4, Stats.minIndex(dataset4));
    }


    @Test
    void minIndicesTestCase() {
        int[] minIndices = {14, 7, 2};
        assertArrayEquals(minIndices, Stats.minIndices(dataset1, 3));
    }


    @Test // Defines a test method
    @DisplayName("Test for sst=0 in determination") // define the name of the test which is displayed to the user
    void determinationExceptionTestCase() {
        assertThrows(Exception.class, () -> Stats.determination(dataset4, dataset1));
    }
}
