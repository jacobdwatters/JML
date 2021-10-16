package com.jml.preprocessing;

import java.util.HashMap;
import java.util.Map;


/**
 * Contains methods to encode classes to integer values.
 */
public class Encode {

    // Hide default constructor.
    private Encode() {throw new IllegalStateException("Cannot instantiate utility class.");}


    /**
     * Encodes a list of classes as values between 0 t0 (n-1) classes.
     *
     * @param labels Labels of a dataset
     * @return An integer array containing the encoding of each label.
     */
    public static int[] standard(String[] labels) {
        int[] encodedLabels = new int[labels.length];
        Map<String, Integer> encodings = new HashMap<>();

        int classNum = 0;

        for (String label : labels) { // Find unique labels
            if (!encodings.containsKey(label)) {
                encodings.put(label, classNum);
                classNum++;
            }
        }

        for(int i=0; i<labels.length; i++) { // set class for each unique label.
            encodedLabels[i] = encodings.get(labels[i]);
        }

        return encodedLabels;
    }
}
