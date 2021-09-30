package com.jml.core;

import com.jml.linear_models.LinearModelTags;
import com.jml.linear_models.PolynomialRegression;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

class PolyRegFromData extends PolynomialRegression {

    private static Scanner scanner;

    /**
     * Creates a polynomial regression model from data.
     *
     * @return The polynomial regression model specified by the data.
     */
    static Model create(List<String> tags, List<String> contents) {
        PolyRegFromData polyRegModel = new PolyRegFromData();
        polyRegModel.isCompiled = true; // Since we are loading a pretrained model, set to true.
        polyRegModel.isFit = true; // Since we are loading a pretrained model, set to true.

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            String tag = tags.remove(0);
            String content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(LinearModelTags.DEGREE.toString())) {
                polyRegModel.degree = scanner.nextInt();

            } else if(tag.equals(LinearModelTags.NORMALIZE.toString())) {
                polyRegModel.normalization = scanner.nextInt();

            } else if(tag.equals(LinearModelTags.COEFFICIENTS.toString())) {
                String[] coeffStrings = content.split(",");

                polyRegModel.coefficients = Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file.");
            }

            scanner.close();
        }

        polyRegModel.buildDetails();
        Class<PolynomialRegression> clazz = PolynomialRegression.class;

        return clazz.cast(polyRegModel);
    }
}
