package com.jml.core;

import com.jml.clasifiers.LogisticRegression;
import com.jml.linear_models.LinearModelTags;
import linalg.Matrix;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;


class LogRegFromData extends LogisticRegression {
    private static Scanner scanner;

    static Model create(List<String> tags, List<String> contents) {
        LogRegFromData logRegModel = new LogRegFromData();
        logRegModel.isFit = true; // Since we are loading a pretrained model, set to true.

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            String tag = tags.remove(0);
            String content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(LinearModelTags.PARAMETERS.toString())) {
                String[] coeffStrings = content.split(",");

                logRegModel.coefficients = Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray();

                logRegModel.w = new Matrix(logRegModel.coefficients.length, 1);
                logRegModel.w.setCol(logRegModel.coefficients, 0);

            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file.");
            }

            scanner.close();
        }

        logRegModel.buildDetails();
        Class<LogisticRegression> clazz = LogisticRegression.class;

        return clazz.cast(logRegModel);
    }
}
