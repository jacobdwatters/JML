package com.jml.preprocessing;

import com.jml.util.ArrayUtils;

import java.util.*;


/**
 * An encoder object that allows for the categorical encoding of samples as one-hot arrays.
 */
public class OneHotEncoder {

    /* Flag for handling unknown samples. If false, a sample that was not seen during the fit() call
        will cause an error. If true, the sample will be "ignored" and all zeros will be returned. */
    private boolean ignoreUnknown;
    private boolean isFit = false;
    private int size; // size of the one-hot encodings.
    Map<String[], int[]> encodings; // Key is the target sample, the value is the encoding of that sample.
    Map<int[], String[]> invEncodings; // The inverse encodings.

    /**
     * Creates a OneHotEncoder object.
     */
    public OneHotEncoder() {
        this.ignoreUnknown = false;
        encodings = new HashMap<>();
        invEncodings = new HashMap<>();
    }


    /**
     * Creates a OneHotEncoder object.
     * @param ignoreUnknown Flag for handling unknown samples. If false, a sample passed to {@link #encode(String[][])} that was not seen
     *                      during the {@link #fit(String[][])} call will cause an error. If true, the sample will be "ignored" and all
     *                      zeros will be returned.
     */
    public OneHotEncoder(boolean ignoreUnknown) {
        this.ignoreUnknown = ignoreUnknown;
        encodings = new HashMap<>();
        invEncodings = new HashMap<>();
    }


    /**
     * Fits the one-hot encoder to a dataset. This creates the encodings for each sample in the targets array. Note that
     * the targets will be sorted by the first column before being encoded.
     * @param targets
     */
    public void fit(String[][] targets) {
        String[][] sortedTargets = ArrayUtils.sortByCol(targets); // Sort, alphanumerically, by the first column.
        List<String[]> toEncode = new ArrayList<>();
        int[] tempEncoding;

        for(String[] sample : sortedTargets) {
            if(!containsArray(toEncode, sample)) {
                toEncode.add(sample); // Then we have a new sample to encode.
            } // Otherwise, we skip.
        }

        size = toEncode.size();

        // Form the one-hot vectors for each sample in the toEncode array.
        for(int i=0; i<toEncode.size(); i++) {
            tempEncoding = new int[toEncode.size()];
            tempEncoding[i] = 1;

            // Create the encoding and inverse encoding.
            encodings.put(toEncode.get(i), tempEncoding);
            invEncodings.put(tempEncoding, toEncode.get(i));
        }

        isFit = true;
    }


    /**
     * Encodes a set of targets to one-hot vectors. OneHotEncoder instance must have already been {@link #fit(String[][]) fit}.
     * <br>
     * - If ignoreUnknown = false, then an error will occur is a sample in targets was not seen during {@link #fit(String[][]) fit}.<br>
     * - If ignoreUnknown = true, and a sample in targets was not seen during the {@link #fit(String[][]) fit} then it will be encoded as all zeros.
     *
     * @param targets A 2D array of strings where each row contains the targets for a single sample.
     * @return One-hot encodings of the specified targets.
     * @throws IllegalStateException If the {@link #fit(String[][])} method has not been called. Or if a sample that was not seen
     * in the {@link #fit(String[][])} method is
     */
    public int[][] encode(String[][] targets) {
        if(!isFit) {
            throw new IllegalStateException("Encoder must be fit before the encode method can be called.");
        }

        int[][] encode = new int[targets.length][size];

        for(int i=0; i<targets.length; i++) {
//            System.out.println("Sample: " + Arrays.toString(targets[i]));
//            System.out.println("Contains: " + containsArrayKey(encodings, targets[i]) + "\n\n");

            if(containsArrayKey(encodings, targets[i])) {
                // Then we have a known encoding for this sample.
                encode[i] = getArray(encodings, targets[i]);

            } else if(ignoreUnknown) {
                // Then the target was not seen in the fit() method, but we ignore the sample.
                encode[i] = new int[size];

            } else {
                throw new IllegalStateException("Could not encode " + targets[i] + " since it was not seen during the fit " +
                        "and ignoreUnknown was set to false.");
            }
        }

        return encode;
    }


    /**
     * Decodes a set of one-hot vectors to their original feature. OneHotEncoder instance must have already been fit.
     * @param onehot A 2D array of strings where each row contains the targets for a single sample.
     * @return One-hot encodings of the specified targets.
     */
    public String[][] decode(int[][] onehot) {
        if(!isFit) {
            throw new IllegalStateException("Encoder must be fit before the decode method can be called.");
        }

        String[][] decode = new String[onehot.length][onehot[0].length];

        for(int i=0; i< onehot.length; i++) {
            decode[i] = getArray(invEncodings, onehot[i]);
        }

        return decode;
    }


    /**
     * Checks if a list of arrays contains a specified array.
     * @param list List to check.
     * @param arr Array of interest.
     * @return True if the list contains the array at least once. False otherwise.
     */
    private boolean containsArray(List<String[]> list, String[] arr) {
        boolean contained = false;

        for(int i=0; i< list.size(); i++) {
            if(Arrays.equals(list.get(i), arr)) {
                contained = true;
                break; // Then we are done.
            }
        }

        return contained;
    }


    /**
     * Checks if a map of arrays contains a specified array as a key.
     * @param map List to check.
     * @param arr Array of interest.
     * @return True if the map contains the array as a key. False otherwise.
     */
    private boolean containsArrayKey(Map<String[], int[]> map, String[] arr) {
        boolean contained = false;

        for(String[] key : map.keySet()) {

            if(Arrays.equals(key, arr)) {
                contained = true;
                break; // Then we are done.
            }
        }

        return contained;
    }


    private int[] getArray(Map<String[], int[]> map, String[] key) {
        int[] arr = null;

        // TODO: Since we are searching the map, there is probably no point in using a map for this...
        //      Could a new java.util.Map be implemented to fix the issue of array equality?
        for(String[] k : map.keySet()) {
            if(Arrays.equals(k, key)) {
                arr = map.get(k);
                break;
            }
        }

        return arr;
    }


    private String[] getArray(Map<int[], String[]> map, int[] key) {
        String[] arr = null;

        // TODO: Since we are searching the map, there is probably no point in using a map for this...
        //      Could a new java.util.Map be implemented to fix the issue of array equality?
        for(int[] k : map.keySet()) {
            if(Arrays.equals(k, key)) {
                arr = map.get(k);
                break;
            }
        }

        return arr;
    }


    // TODO: TEMP. Remove
    public void printMap(Map<String[], int[]> map) {
        for(String[] key : map.keySet()) {
            System.out.println("key: " + Arrays.toString(key) + ", value: " + Arrays.toString(map.get(key)));
        }
    }


    // TODO: TEMP. Remove
    public static void main(String[] args) {
        String[][] data = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}};

        OneHotEncoder encoder = new OneHotEncoder(true);
        encoder.fit(data);

        int[][] encodings = encoder.encode(data);
        String[][] decodings = encoder.decode(encodings);

        System.out.println("Encodings");
        for(int[] onehot : encodings) {
            System.out.println(Arrays.toString(onehot));
        }


        System.out.println("\n\nDecodings:");
        for(String[] target : decodings) {
            System.out.println(Arrays.toString(target));
        }
    }
}
