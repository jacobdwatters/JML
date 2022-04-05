package com.jml.preprocessing;

import com.jml.util.ArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


/**
 * Encodes a list of classes as numerical values between 0 to (n-1) for n classes.<br>
 * This encoder should be used to encode the target values of a dataset and not the features.
 */
public class ClassEncoder implements Encoder {
    boolean ignoreUnknown;
    boolean isFit;
    Map<String, Integer> encodings; // Class encodings.
    Map<Integer, String> invEncodings; // Inverse encodings.


    /**
     * Creates a class encoder. Note, by default, this will not ignore unknown samples. See {@link #ClassEncoder(boolean)}
     * to change this.
     */
    public ClassEncoder() {
        encodings = new HashMap<>();
        invEncodings = new HashMap<>();
        ignoreUnknown = false;
        isFit = false;
    }


    /**
     * Creates a class encoder.
     * @param ignoreUnknown Flag for whether to ignore unseen samples. If true, then samples not seen in {@link #fit(String[][])}
     *                      will be ignored. If false, then when trying to encode a sample not seen in
     *                      {@link #fit(String[][])} an exception will be thrown.
     */
    public ClassEncoder(boolean ignoreUnknown) {
        encodings = new HashMap<>();
        invEncodings = new HashMap<>();
        this.ignoreUnknown = ignoreUnknown;
        isFit = false;
    }


    /**
     * Encodes targets as numerical class labels. Note, this fit method will flatten the targets array and create encodings with that.
     * @param targets Targets to encode as numerical class labels.
     */
    @Override
    public void fit(String[][] targets) {
        String[] sortedLabels = ArrayUtils.flatten(targets);

        Arrays.sort(sortedLabels);

        int classNum = 0;

        for (String label : sortedLabels) {

            // Find unique labels
            if (!encodings.containsKey(label)) {

                // Store the encoding.
                encodings.put(label, classNum);
                invEncodings.put(classNum, label);
                classNum++;
            }
        }

        isFit = true;
    }


    /**
     * Encodes a set of targets as numerical integer labels.
     * @param targets Targets to encode.
     * @return An array of the same shape as targets containing the numerical encodings of the targets.
     * @throws IllegalStateException If the {@link #fit(String[][])} method has not been called. Or if a sample that was not seen
     * in the {@link #fit(String[][])} method and ignoreUnknown is set to false.
     */
    @Override
    public int[][] encode(String[][] targets) {
        if(!isFit) {
            throw new IllegalStateException("Encoder must be fit before targets can be encoded.");
        }

        int[][] encode = new int[targets.length][targets[0].length];

        for(int i=0; i<targets.length; i++) {
            for(int j=0; j<targets[0].length; j++) {
                if(encodings.containsKey(targets[i][j])) {
                    // Then encode as normal
                    encode[i][j] = encodings.get(targets[i][j]);

                } else if(ignoreUnknown) {
                    // Then the target was not seen in the fit() method, but we ignore the sample.
                    encode[i][j] = -1;

                } else {
                    throw new IllegalStateException("Could not encode " + targets[i][j] + " since it was not seen during the fit\n" +
                            "and ignoreUnknown was set to false.");
                }
            }
        }

        return encode;
    }


    /**
     * Decodes a set of numerical classes into their string representations. If there is a class in the classes array
     * which was not seen in the {@link #fit(String[][])} array, then null will be assigned.
     * @param classes Classes to decode.
     * @return An array of the same shape as classes containing the string decodings of each class value.
     */
    @Override
    public String[][] decode(int[][] classes) {
        if(!isFit) {
            throw new IllegalStateException("Encoder must be fit before classes can be decoded.");
        }

        String[][] decode = new String[classes.length][classes[0].length];

        for(int i=0; i<classes.length; i++) {
            for(int j=0; j<classes[0].length; j++) {
                decode[i][j] = invEncodings.get(classes[i][j]);
            }
        }

        return decode;
    }
}
