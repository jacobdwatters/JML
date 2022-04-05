package com.jml.preprocessing;


/**
 * An interface for specifying data encoders.<br>
 * Included default encoders:
 * <pre>
 *     - {@link OneHotEncoder}
 *     - {@link ClassEncoder}</pre>
 */
public interface Encoder {

    /**
     * Fits the encoder to the data. This generates an encodings for each unique sample.
     * @param data Data to generate encodings for.
     */
    void fit(String[][] data);


    /**
     * Encodes samples based on the encodings generated in the {@link #fit(String[][])} method.
     * @param samples Samples to encode.
     * @return The encodings of each sample in the samples array.
     * @throws IllegalStateException if this method is called before {@link #fit(String[][])}.
     */
    int[][] encode(String[][] samples);


    /**
     * Decodes samples based on the encodings generated in the {@link #fit(String[][])} method.
     * @param samples Samples to decode.
     * @return The decodings of each sample in the samples array.
     */
    String[][] decode(int[][] samples);
}
