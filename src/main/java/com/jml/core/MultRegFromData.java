package com.jml.core;

import com.jml.linear_models.LinearModelTags;
import com.jml.linear_models.MultipleLinearRegression;
import linalg.Matrix;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

class MultRegFromData extends MultipleLinearRegression {

    private static Scanner scanner;

    static Model create(List<String> tags, List<String> contents) {
        MultRegFromData multRegModel = new MultRegFromData();
        multRegModel.isFit = true; // Since we are loading a pretrained model, set to true.

        String tag, content;

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            tag = tags.remove(0);
            content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(LinearModelTags.PARAMETERS.toString())) {
                String[] coeffStrings = content.split(",");

                multRegModel.coefficients = Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

                multRegModel.w = new Matrix(multRegModel.coefficients.length, 1);
                multRegModel.w.setCol(multRegModel.coefficients, 0);

            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file: " + tag);
            }

            scanner.close();
        }

        multRegModel.buildDetails();
        Class<MultipleLinearRegression> clazz = MultipleLinearRegression.class;

        return clazz.cast(multRegModel);
    }
}
