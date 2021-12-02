package com.jml.classifiers;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

class KDTreeTest {

    double[][] values;
    KDTree tree;
    int k;


    @Test
    void kd2TreeTest() {
        double[][] values = {{7, 2},
                            {5, 4},
                            {9, 6},
                            {4, 7},
                            {8, 1},
                            {2, 3}};
        k=2;
        tree = new KDTree(k);

        double[][] expected = { {2.0, 3.0, },
                                {5.0, 4.0, },
                                {4.0, 7.0, },
                                {7.0, 2.0, },
                                {8.0, 1.0, },
                                {9.0, 6.0}};

        for(double[] point : values) {
            tree.insert(point);
        }

        List<double[]> result = tree.inOrder();
        double[][] actual = new double[values.length][k];

        for(int i=0; i<result.size(); i++) {
            for(int j=0; j<k; j++) {
                actual[i][j] = result.get(i)[j];
            }
        }

        assertArrayEquals(expected, actual);
    }


    @Test
    void kd3TreeTest() {
        double[][] values = {   {1, 3, 5},
                                {5, 6, 1},
                                {5, 5, 2},
                                {10, -2, 0},
                                {0, 1, 3.4},
                                {2, 3, 4}};
        k=3;
        tree = new KDTree(k);

        double[][] expected = { {0.0, 1.0, 3.4, },
                                {1.0, 3.0, 5.0, },
                                {10.0, -2.0, 0.0, },
                                {5.0, 5.0, 2.0, },
                                {2.0, 3.0, 4.0, },
                                {5.0, 6.0, 1.0}};

        for(double[] point : values) {
            tree.insert(point);
        }

        List<double[]> result = tree.inOrder();
        double[][] actual = new double[values.length][k];

        for(int i=0; i<result.size(); i++) {
            for(int j=0; j<k; j++) {
                actual[i][j] = result.get(i)[j];
            }
        }

        assertArrayEquals(expected, actual);
    }

    @Test
    void kdTreeExceptionTest() {
        tree = new KDTree(4);

        // Attempt to insert points of wrong dimension.
        assertThrows(Exception.class, () -> tree.insert(new double[]{1, 2, 3}));
        assertThrows(Exception.class, () -> tree.insert(new double[]{1, 2, 3, 4, 5}));
        assertThrows(Exception.class, () -> tree.insert(new double[]{}));

        // Attempt to construct kd tree with negative dimension points
        assertThrows(Exception.class, () -> new KDTree(-1));
        assertThrows(Exception.class, () -> new KDTree(0));
        assertThrows(Exception.class, () -> new KDTree(-2));
    }

}
