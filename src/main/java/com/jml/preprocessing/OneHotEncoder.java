package com.jml.preprocessing;

import java.util.Map;


/**
 * An object that allows for the categorical encoding of samples as one-hot arrays.
 */
public class OneHotEncoder {

    /* Flag for handling unknown samples. If false, a sample that was not seen during the fit() call
        will cause an error. If true, the sample will be "ignored" and all zeros will be returned. */
    private boolean ignoreUnknown;
    private boolean isFit = false;
    Map<String[], int[]> encodings; // Key is the target sample, the value is the encoding of that sample.
    Map<int[], String[]> invEncodings; // The inverse encodings.

    /**
     * Creates a OneHotEncoder object.
     */
    public OneHotEncoder() {
        this.ignoreUnknown = false;
    }


    /**
     * Creates a OneHotEncoder object.
     * @param ignoreUnknown Flag for handling unknown samples. If false, a sample passed to encode() that was not seen
     *                      during the fit() call will cause an error. If true, the sample will be "ignored" and all
     *                      zeros will be returned.
     */
    public OneHotEncoder(boolean ignoreUnknown) {
        this.ignoreUnknown = ignoreUnknown;
    }


    public void fit(String[][] targets) {
        // TODO:

        isFit = true;
    }


    /**
     * Encodes a set of targets to one-hot vectors. OneHotEncoder instance must have already been fit.
     * @param targets A 2D array of strings where each row contains the targets for a single sample.
     * @return One-hot encodings of the specified targets.
     */
    public int[][] encode(String[][] targets) {
        // TODO: Implement ignoreUnknown
        if(!isFit) {
            throw new IllegalStateException("Encoder must be fit before the encode method can be called.");
        }

        int[][] encode = new int[targets.length][targets[0].length];

        for(int i=0; i< targets.length; i++) {
            encode[i] = encodings.get(targets[i]);
        }

        return encode;
    }


    /**
     * Decodes a set of one-hot vectors to their original feature. OneHotEncoder instance must have already been fit.
     * @param onehot A 2D array of strings where each row contains the targets for a single sample.
     * @return One-hot encodings of the specified targets.
     */
    public String[][] decode(int[][] onehot) {
        // TODO: Implement ignoreUnknown
        if(!isFit) {
            throw new IllegalStateException("Encoder must be fit before the decode method can be called.");
        }

        String[][] encode = new String[onehot.length][onehot[0].length];

        for(int i=0; i< onehot.length; i++) {
            encode[i] = invEncodings.get(onehot[i]);
        }

        return encode;
    }
}
