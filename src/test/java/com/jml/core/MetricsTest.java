package com.jml.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetricsTest {

    @Test
    void accuracy1dTest() {
        double[] y1 =      {1, 4,  5, 0, 1, 6, -12.1,  323.4, 12,  0};
        double[] y1_pred = {1, 3, 42, 5, 1, 2, -12, 323.40, 12, -0};

        double[] y2 = {};
        double[] y2_pred = {};

        double[] y3 = {-0.31, 4, 1};
        double[] y3_pred = {-0.31, 3, 1};

        double[] y4 = {1, 2, 3, 4, -12};
        double[] y4_pred = {1, 2, 3, 4, -12};

        double acu1 = 0.5, acu3 = 0.6666666666666666;

        assertEquals(acu1, Metrics.accuracy(y1_pred, y1));
        assertTrue(Double.isNaN(Metrics.accuracy(y2_pred, y2)));
        assertEquals(acu3, Metrics.accuracy(y3_pred, y3));

        assertEquals(1, Metrics.accuracy(y4_pred, y4));

        assertThrows(IllegalArgumentException.class, () -> Metrics.accuracy(y1, y3_pred));
    }

    @Test
    void accuracy2dTest() {
        double[][] y1 = {   {1, 5, 1},
                            {0, 12.2, -1},
                            {0, 13, 1}};
        double[][] y1_pred ={   {1, 5, 1},
                                {0, 1, -1},
                                {0, 13, -1}};

        double[][] y2 = {   {1, 2, 1},
                            {4, 5, 1},
                            {-9, 2, 1},
                            {7, 1, 1}};
        double[][] y2_pred= {   {1, 2, 1},
                                {4, 5, 1},
                                {-9, 2, 1},
                                {7, 34.5, 1}};
        double[][] y3 = {   {1},
                            {3},
                            {-1.2}};
        double[][] y3_pred  = { {1},
                                {3},
                                {-1.4}};

        double[][] y4 = {{1, 2, -1},
                {4, 5, 6},
                {-10.3, 3, 0}};
        double[][] y4_pred = {{1, 2, -1},
                {4, 5, 6},
                {-10.3, 3, 0}};

        double acu1 = 0.3333333333333333, acu2 = 0.75, acu3 = 0.6666666666666666;

        assertEquals(acu1, Metrics.accuracy(y1_pred, y1));
        assertEquals(acu2, Metrics.accuracy(y2_pred, y2));
        assertEquals(acu3, Metrics.accuracy(y3_pred, y3));

        assertEquals(1, Metrics.accuracy(y4_pred, y4));

        assertThrows(IllegalArgumentException.class, () -> Metrics.accuracy(y1, y2_pred));
        assertThrows(IllegalArgumentException.class, () -> Metrics.accuracy(y1, y3_pred));
        assertThrows(IllegalArgumentException.class, () -> Metrics.accuracy(y2, y3_pred));
    }
}
