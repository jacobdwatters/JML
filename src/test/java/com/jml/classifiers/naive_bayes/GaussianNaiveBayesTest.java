package com.jml.classifiers.naive_bayes;

import com.jml.core.Model;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GaussianNaiveBayesTest {

    GaussianNaiveBayes gnb;

    double[][] X1 = {{-1, -1}, {-2, -1}, {-3, -2}, {1, 1}, {2, 1}, {3, 2}};
    double[] y1 = {1, 1, 1, 2, 2, 2};
    double[][] t1 = {{-0.8, -1}, {2, 1}};

    double[][] X2 = {{1}, {4}, {-1}, {0.612}, {1.2}, {1.234}, {-8.123}, {12}, {-5.1}, {3}};
    double[] y2 = {1, 1, 0, 0, 1, 3, 1, 4, 1, 2};
    double[][] t2 = {{3}, {-0.4}, {8}};

    @BeforeEach
    void setup() {
        gnb = new GaussianNaiveBayes();
    }


    @Test
    void dataset1Test() {
        gnb.fit(X1, y1);
        double[] y1pred = gnb.predict(X1);
        double[] t1pred = gnb.predict(t1);

        assertArrayEquals(new double[]{1, 1, 1, 2, 2, 2}, y1pred);
        assertArrayEquals(new double[]{1, 2}, t1pred);

        gnb.saveModel("./src/test/java/com/jml/classifiers/naive_bayes/testfiles/gnbTest1.mdl");
        Model<double[][], double[]> gnbLoad = Model.load("./src/test/java/com/jml/classifiers/naive_bayes/testfiles/gnbTest1.mdl");

        double[] y1predLoad = gnbLoad.predict(X1);
        double[] t1predLoad = gnbLoad.predict(t1);
        assertArrayEquals(new double[]{1, 1, 1, 2, 2, 2}, y1predLoad);
        assertArrayEquals(new double[]{1, 2}, t1predLoad);
    }


    @Test
    void dataset2Test() {
        gnb.fit(X2, y2);
        double[] y2pred = gnb.predict(X2);
        double[] t2pred = gnb.predict(t2);

        assertArrayEquals(new double[]{1, 1, 0, 0, 1, 3, 1, 4, 1, 2}, y2pred);
        assertArrayEquals(new double[]{2, 0, 1}, t2pred);

        gnb.saveModel("./src/test/java/com/jml/classifiers/naive_bayes/testfiles/gnbTest2.mdl");
        Model<double[][], double[]> gnbLoad = Model.load("./src/test/java/com/jml/classifiers/naive_bayes/testfiles/gnbTest2.mdl");

        double[] y1predLoad = gnbLoad.predict(X2);
        double[] t1predLoad = gnbLoad.predict(t2);
        assertArrayEquals(new double[]{1, 1, 0, 0, 1, 3, 1, 4, 1, 2}, y1predLoad);
        assertArrayEquals(new double[]{2, 0, 1}, t1predLoad);
    }
}
