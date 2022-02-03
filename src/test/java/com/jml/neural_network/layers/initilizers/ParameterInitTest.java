package com.jml.neural_network.layers.initilizers;

import com.jml.core.Stats;
import linalg.Matrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParameterInitTest {

    int m;
    int n;

    @BeforeEach
    void setUp () {
        m = 16;
        n = 21;
    }


    @RepeatedTest(5)
    void randNormTest() {
        double std = 1;
        double mean = 0;
        RandomNormal init = new RandomNormal();
        Matrix rand = init.init(m, n);

        double dataMean = Stats.mean(rand.flatten().getValuesAsDouble()[0]);
        double dataStd = Stats.std(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertEquals(init.mean, mean);
        assertEquals(init.std, std);
        assertEquals(Stats.round(dataMean, 0), mean);
        assertEquals(Stats.round(dataStd, 0), std);
    }


    @RepeatedTest(5)
    void randNorm2Test() {
        double std = 2.5;
        double mean = 3;
        RandomNormal init = new RandomNormal(std, mean);
        Matrix rand = init.init(m, n);

        double dataMean = Stats.mean(rand.flatten().getValuesAsDouble()[0]);
        double dataStd = Stats.std(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertEquals(init.mean, mean);
        assertEquals(init.std, std);
        assertEquals(Stats.round(dataMean, 0), mean);
        assertTrue(Math.abs(std-dataStd) < 1); // Check that std within 1
    }


    @RepeatedTest(5)
    void randUniTest() {
        double min = 0;
        double max = 1;
        RandomUniform init = new RandomUniform();
        Matrix rand = init.init(m, n);

        double dataMin = Stats.min(rand.flatten().getValuesAsDouble()[0]);
        double dataMax = Stats.max(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertEquals(init.min, min);
        assertEquals(init.max, max);
        assertTrue(min <= dataMin);
        assertTrue(max >= dataMax);
    }


    @RepeatedTest(5)
    void randUni2Test() {
        double min = -1;
        double max = 5.9;
        RandomUniform init = new RandomUniform(min, max);
        Matrix rand = init.init(m, n);

        double dataMin = Stats.min(rand.flatten().getValuesAsDouble()[0]);
        double dataMax = Stats.max(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertEquals(init.min, min);
        assertEquals(init.max, max);
        assertTrue(min <= dataMin);
        assertTrue(max >= dataMax);
    }


    @Test
    void constTest() {
        Constant init = new Constant(3);
        Matrix rand = init.init(m, n);
        Matrix act = new Matrix(m, n, 3);

        assertArrayEquals(rand.getValuesAsDouble(), act.getValuesAsDouble());
        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
    }


    @Test
    void constDefaultTest() {
        Constant init = new Constant();
        Matrix rand = init.init(m, n);
        Matrix act = Matrix.ones(m, n);

        assertArrayEquals(rand.getValuesAsDouble(), act.getValuesAsDouble());
        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
    }


    @Test
    void onesTest() {
        Ones init = new Ones();
        Matrix rand = init.init(m, n);
        Matrix act = Matrix.ones(m, n);

        assertArrayEquals(rand.getValuesAsDouble(), act.getValuesAsDouble());
        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
    }


    @Test
    void zerosTest() {
        Zeros init = new Zeros();
        Matrix rand = init.init(m, n);
        Matrix act = Matrix.zeros(m, n);

        assertArrayEquals(rand.getValuesAsDouble(), act.getValuesAsDouble());
        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
    }


    @RepeatedTest(5)
    void glorotNormTest() {
        double std = Math.sqrt(6.0/(m+n));
        double mean = 0;
        GlorotNormal init = new GlorotNormal();
        Matrix rand = init.init(m, n);

        double dataMean = Stats.mean(rand.flatten().getValuesAsDouble()[0]);
        double dataStd = Stats.std(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertEquals(Stats.round(dataMean, 0), mean);
        assertTrue(Math.abs(std-dataStd) < 1); // Check that std within 1
    }


    @RepeatedTest(5)
    void glorotUniTest() {
        double min = -Math.sqrt(6.0/(m+n));
        double max = Math.sqrt(6.0/(m+n));
        GlorotUniform init = new GlorotUniform();
        Matrix rand = init.init(m, n);

        double dataMin = Stats.min(rand.flatten().getValuesAsDouble()[0]);
        double dataMax = Stats.max(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertTrue(min <= dataMin);
        assertTrue(max >= dataMax);
    }


    @RepeatedTest(5)
    void heNormTest() {
        double std = Math.sqrt(2.0/(n));
        double mean = 0;
        HeNormal init = new HeNormal();
        Matrix rand = init.init(m, n);

        double dataMean = Stats.mean(rand.flatten().getValuesAsDouble()[0]);
        double dataStd = Stats.std(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertEquals(Stats.round(dataMean, 0), mean);
        assertTrue(Math.abs(std-dataStd) < 1); // Check that std within 1
    }


    @RepeatedTest(5)
    void heUniTest() {
        double min = -Math.sqrt(2.0/(n));
        double max = Math.sqrt(2.0/(n));
        HeUniform init = new HeUniform();
        Matrix rand = init.init(m, n);

        double dataMin = Stats.min(rand.flatten().getValuesAsDouble()[0]);
        double dataMax = Stats.max(rand.flatten().getValuesAsDouble()[0]);

        assertEquals(rand.numRows(), m);
        assertEquals(rand.numCols(), n);
        assertTrue(min <= dataMin);
        assertTrue(max >= dataMax);
    }



}
