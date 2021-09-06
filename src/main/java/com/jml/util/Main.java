package com.jml.util;


import com.jml.neural_network.activations.Sigmoid;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Layer;

public class Main {
    public static void main(String[] args) {
        Layer l = new Dense(1, 1, "sigmoid");
    }
}
