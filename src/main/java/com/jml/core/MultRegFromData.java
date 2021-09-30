package com.jml.core;

import com.jml.linear_models.LinearModelTags;
import com.jml.linear_models.MultipleLinearRegression;
import com.jml.util.ArrayUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

class MultRegFromData extends MultipleLinearRegression {

    private static Scanner scanner;

    static Model create(List<String> tags, List<String> contents) {
        MultRegFromData multRegModel = new MultRegFromData();
        multRegModel.isCompiled = true; // Since we are loading a pretrained model, set to true.
        multRegModel.isFit = true; // Since we are loading a pretrained model, set to true.

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            String tag = tags.remove(0);
            String content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(LinearModelTags.NORMALIZE.toString())) {
                multRegModel.normalization = scanner.nextInt();

            } else if(tag.equals(LinearModelTags.COEFFICIENTS.toString())) {
                String[] coeffStrings = content.split(",");

                multRegModel.coefficients = Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file.");
            }

            scanner.close();
        }

        multRegModel.buildDetails();
        Class<MultipleLinearRegression> clazz = MultipleLinearRegression.class;

        return clazz.cast(multRegModel);
    }
}
