package com.jml.core;

import com.jml.linear_models.LinearModelTags;
import com.jml.linear_models.LinearRegression;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

class LinRegFromData extends LinearRegression {
    private static Scanner scanner;

    /**
     * Creates a polynomial regression model from data.
     *
     * @return The polynomial regression model specified by the data.
     */
    static Model create(List<String> tags, List<String> contents) {
        LinRegFromData linRegModel = new LinRegFromData();
        linRegModel.isCompiled = true; // Since we are loading a pretrained model, set to true.
        linRegModel.isFit = true; // Since we are loading a pretrained model, set to true.

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            String tag = tags.remove(0);
            String content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(LinearModelTags.NORMALIZE.toString())) {
                linRegModel.normalization = scanner.nextInt();

            } else if(tag.equals(LinearModelTags.COEFFICIENTS.toString())) {
                String[] coeffStrings = content.split(",");

                linRegModel.coefficients = Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file.");
            }

            scanner.close();
        }

        linRegModel.buildDetails();
        Class<LinearRegression> clazz = LinearRegression.class;

        return clazz.cast(linRegModel);
    }
}
