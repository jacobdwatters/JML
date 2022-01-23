package com.jml.neural_network.layers;

import linalg.Matrix;

import java.util.Random;

public class Initializers {
    // TODO: Make initializers a private interface with one init(int m, int n) method. randn, randUniform etc. will extend it.

    static Random r = new Random();

    static Matrix randn(int m, int n, double std) {
        Matrix random = new Matrix(m, n);

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                random.set(r.nextGaussian()*std, i, j);
            }
        }

        return random;
    }
}
