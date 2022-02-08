package com.jml.core;

import com.jml.classifiers.naive_bayes.GaussianNaiveBayes;
import com.jml.linear_models.LinearModelTags;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * Gaussian Naive Bayes model from data.
 */
class GnbFromData extends GaussianNaiveBayes {
    private static Scanner scanner;

    static Model create(List<String> tags, List<String> contents) {
        GnbFromData  gnbModel = new GnbFromData ();
        gnbModel.isFit = true; // Since we are loading a pretrained model, set to true.

        String tag, content;

        double[][] features = null;
        double[] targets = null;

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            tag = tags.remove(0);
            content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(LinearModelTags.FEATURES.toString())) {
                // Form the weight matrix for the layer.
                String[] rows = content.split(";");
                int rowLength = rows[0].split(",").length;
                features = new double[rows.length][rowLength];

                for(int i=0; i<features.length; i++) {
                    String[] rowVals = rows[i].split(",");

                    for(int j=0; j<features[0].length; j++) {
                        features[i][j] = Double.parseDouble(rowVals[j]);
                    }
                }


            } else if(tag.equals(LinearModelTags.TARGETS.toString())) {
                String[] coeffStrings = content.split(",");

                targets= Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file: " + tag);
            }

            scanner.close();
        }

        gnbModel.fit(features, targets);

        gnbModel.buildInspection();
        Class<GaussianNaiveBayes> clazz = GaussianNaiveBayes.class;

        return clazz.cast(gnbModel);
    }
}
