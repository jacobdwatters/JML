package com.jml.preprocessing;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


/**
 * Contains methods to encode classes or targets to numerical values.
 */
public class Encoder {

    // Hide default constructor.
    private Encoder() {throw new IllegalStateException("Cannot instantiate utility class.");}

    // TODO: Add Encoding object that hold a map which shows what class became what value. That way the value can be decoded.


    /**
     * Encodes a list of classes as values between 0 t0 (n-1) classes. <br><br>
     * Labels will be sorted alphabetically before encoding. This guarantees a consistent method of encoding so that
     * the encoded labels can be decoded.
     *
     * @param labels Labels of a dataset.
     * @return An integer array containing the encoding of each label.
     */
    public static int[] encodeClasses(int[] labels) {
        String[] labelsAsStr = Arrays.toString(labels)
                .replaceAll("\\s+", "")
                .split(",");
        return encodeClasses(labelsAsStr);
    }


    /**
     * Encodes a list of classes as values between 0 t0 (n-1) classes.<br><br>
     * Labels will be sorted alphabetically before encoding. This guarantees a consistent method of encoding so that
     * the encoded labels can be decoded.
     *
     * @param labels Labels of a dataset.
     * @return An integer array containing the encoding of each label.
     */
    public static int[] encodeClasses(String[] labels) {
        int[] encodedLabels = new int[labels.length];
        Map<String, Integer> encodings = new HashMap<>();
        Arrays.sort(labels);

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


    /**
     * Encodes a list of classes as values between 0 t0 (n-1) classes.<br>
     * WARNING: This method only flattens the array and calls {@link #encodeClasses(String[])}.<br><br>
     *
     * Labels will be sorted alphabetically before encoding. This guarantees a consistent method of encoding so that
     * the encoded labels can be decoded.
     *
     * @param labels Labels of a dataset.
     * @return An integer array containing the encoding of each label.
     */
    public static int[] encodeClasses(String[][] labels) {
        String[] flat = new String[labels.length* labels[0].length];
        int k=0;

        for(int i=0; i<labels.length; i++) { // Flatten the array.
            for(int j=0; j<labels[0].length; j++) {
                flat[k] = labels[i][j];
                k++;
            }
        }

        return encodeClasses(flat);
    }
}
