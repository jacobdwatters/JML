package com.jml.preprocessing;

import java.util.*;

/**
 * Base data structure for the {@link OneHotEncoder one-hot encoder}. This stores the one-hot encodings for a single
 * feature of a dataset. This object acts as a one-hot encoder, for datasets with only a single feature.
 */
class OneHotBase {
    Map<String, int[]> encodings;


    /**
     * Creates a OneHotBase object where ignoreUnknown is set to false.
     */
    public OneHotBase() {
        encodings = new HashMap<>();
    }


    /**
     * Computes the onehot vector for each feature.
     * @param features Features to encode as one-hot vectors.
     */
    public void fit(String[] features) {
        List<String> toEncode = new ArrayList<>();
        String[] sortedFeatures = features.clone();
        Arrays.sort(sortedFeatures);
        int[] tempEncoding;

        for(String sample : sortedFeatures) {

            if(!toEncode.contains(sample)) {
                toEncode.add(sample); // Then we have a new sample to encode.
            } // Otherwise, we skip.
        }

        for(int i=0; i<toEncode.size(); i++) {
            tempEncoding = new int[toEncode.size()];
            tempEncoding[i] = 1;

            // Create the encoding and inverse encoding.
            encodings.put(toEncode.get(i), tempEncoding);
        }
    }
}
