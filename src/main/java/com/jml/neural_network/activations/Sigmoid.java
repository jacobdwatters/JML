package com.jml.neural_network.activations;

public class Sigmoid extends Activation<double[]> {


    /**
     * Applies the sigmoid activation function to the data.<br>
     * Specifically, the logistic function is applied. i.e. 1/(1+e<sup>-x</sup>)
     *
     * @param data Data to apply the activation function to.
     * @return The image of the data under the activation function.
     */
    @Override
    public double[] apply(double[] data) {
        double[] result = new double[data.length];

        for(int i=0; i<data.length; i++) {
            result[i] = 1 / (data[i]);
        }

        return result;
    }
}
