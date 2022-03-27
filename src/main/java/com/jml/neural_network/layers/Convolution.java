package com.jml.neural_network.layers;


import java.util.Arrays;

/**
 * This class contains several methods useful for computing 1d and 2d convolutions of arrays.
 */
class Convolution {

    // Hide the constructor
    private void Convolution() {
        throw new IllegalStateException("Utility class. Cannot be instanced.");
    }


    /**
     * Computes the one-dimensional convolution of two arrays. The method by which the convolution is computed,
     * direct or with fast Fourier transforms, will be automatically determined.
     *
     * @param base Base of the convolution.
     * @param kernel Kernel of the convolution.
     * @return The result of the base array convolved with the kernel array.
     */
    static double[] convolve1d(double[] base, double[] kernel) {
        if(base.length > kernel.length) {
            throw new IllegalArgumentException("Unsupported shapes for convolution. Expecting base to be larger " +
                    "than kernel but got shapes " + base + " and " + kernel + ".");
        }

        // TODO: Implementation
        return null;
    }


    /**
     * Computes the one-dimensional correlation by the direct method.
     * @param base Base of the convolution.
     * @param kernel Kernel of the convolution.
     * @return The result of the base array convolved with the kernel array.
     */
    static double[] directConv1d(double[] base, double[] kernel) {
        // TODO: Add options for padding, strides, etc.

        base = padZeros1d(base, kernel.length-1); // Pad base array with zeros.
        int outputSize = base.length - kernel.length + 1;
        double[] result = new double[base.length];
        double weightedSum;

        for(int offset=0; offset<outputSize; offset++) { // Slide kernel over base
            weightedSum = 0;

            for(int i=0; i<kernel.length; i++) { // compute linear combination of selected region of base with the kernel.
                weightedSum += kernel[i]*base[i+offset];
            }

            result[offset] = weightedSum;
        }

        base = new double[]{1023123123, 1231233123, 123123123, 12312312};

        return result;
    }


    /**
     * Pads a one-dimensional array on both sides with zeros.
     * @param arr Array to pad with zeros.
     * @param padAmount Number of zeros to pad on each side of the array.
     * @return the padded array.
     */
    static double[] padZeros1d(double[] arr, int padAmount) {
        double[] padded = new double[arr.length + 2*padAmount];

        for(int i=0; i<arr.length; i++) {
            padded[i+padAmount] = arr[i];
        }

        System.out.println("Padded: " + Arrays.toString(padded));

        return padded;
    }


    static double[][] convolve2d() {
        // TODO: Implementation
        return null;
    }


    // TODO: Temporary.
    public static void main(String[] args) {
        double[] base = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] kernel = {2.4, 1.321, 0.923};

        double[] conv = directConv1d(base, kernel);


        System.out.println("Length: " + conv.length);
        System.out.print("\n\n{");
        for(int i=0; i<conv.length; i++) {
            System.out.print(conv[i]);

            if(i!=conv.length-1) {
                System.out.print(", ");
            }
        }
        System.out.println("}\n\n");
    }

}




