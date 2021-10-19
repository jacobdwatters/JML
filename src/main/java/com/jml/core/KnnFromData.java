package com.jml.core;

import com.jml.clasifiers.ClassifierTags;
import com.jml.clasifiers.KNearestNeighbors;
import com.jml.linear_models.LinearModelTags;
import com.jml.linear_models.LinearRegression;
import com.jml.util.ArrayUtils;
import linalg.Matrix;
import linalg.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;


class KnnFromData extends KNearestNeighbors {
    private static Scanner scanner;

    /**
     * Creates a polynomial regression model from data.
     *
     * @return The polynomial regression model specified by the data.
     */
    static Model create(List<String> tags, List<String> contents) {
        KnnFromData knnModel = new KnnFromData();
        knnModel.isFit = true; // Since we are loading a pretrained model, set to true.

        while(!tags.isEmpty() && !contents.isEmpty()) {
            // Get the tag/content pair
            String tag = tags.remove(0);
            String content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(ClassifierTags.K.toString())) {
                knnModel.k = scanner.nextInt();

            } else if(tag.equals(ClassifierTags.P.toString())) {
                knnModel.p = scanner.nextInt();

            } else if(tag.equals(ClassifierTags.FEATURES.toString())) {
                List<double[]> features = new ArrayList<>();
                String[] sample = new String[0];

                sample = scanner.nextLine().split(";");

                for(int i=0; i< sample.length; i++) {
                    features.add(Arrays.stream(sample[i].split(","))
                            .mapToDouble(Double::parseDouble)
                            .toArray());
                }

                double[][] featureArray = new double[features.size()][sample.length];

                for(int i=0; i<featureArray.length; i++) {
                    featureArray[i] = features.get(i);
                }

                knnModel.X = new Matrix(featureArray);

            } else if(tag.equals(ClassifierTags.CLASSES.toString())) {
                String[] coeffStrings = content.split(",");

                knnModel.y = new Vector(Arrays.stream(coeffStrings)
                        .mapToDouble(Double::parseDouble)
                        .toArray(), 1);
            } else {
                throw new IllegalArgumentException("Failed to load model. Unrecognized tag in file.");
            }

            scanner.close();
        }

        knnModel.buildDetails();
        Class<KNearestNeighbors> clazz = KNearestNeighbors.class;

        return clazz.cast(knnModel);
    }
}
