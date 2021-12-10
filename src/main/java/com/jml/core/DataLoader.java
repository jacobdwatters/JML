package com.jml.core;

import com.jml.util.FileManager;

import java.util.ArrayList;
import java.util.List;

/**
 * The DataLoader class contains several methods to load data for models.
 */
public class DataLoader {

    private DataLoader() {
        throw new IllegalStateException("Utility class. Cannot be instantiated.");
    }


    /**
     * Loads targets and features from a csv file. <br><br>
     *
     * The csv file is assumed to have one data sample per row. The last
     * column is assumed to be the target column, while all other columns are feature columns. If you wish to specify
     * what column indices contain targets vs features see {@link #loadFeaturesAndTargets(String, int[], int[])}.
     *
     * @param filePath The path, including the extension, of the csv file containing the targets and features.
     * @return An array list of length two containing the features and targets of the dataset in that order.
     */
    public static List<String[][]> loadFeaturesAndTargets(String filePath) {
        List<String[][]> data = new ArrayList<>();
        String[][] content = FileManager.readCSVtoString(filePath);

        String[][] features = new String[content.length][content[0].length-1];
        String[][] targets = new String[content.length][1];

        for(int i=0; i< content.length; i++) {
            for(int j=0; j< content[0].length-1; j++) {
                features[i][j] = content[i][j];
            }

            targets[i][0] = content[i][content[0].length-1];
        }

        data.add(features);
        data.add(targets);

        return data;
    }


    /**
     * Loads targets and features from a csv file.<br><br>
     *
     * The csv file is assumed to have one data sample per row. Each column is either a feature or a target.
     * Also see {@link #loadFeaturesAndTargets(String)}
     *
     * @param filePath The path, including the extension, of the csv file containing the targets and features.
     * @param featureColumns Indices of columns containing features.
     * @param targetColumns Indices of columns containing targets.
     * @return An array list of length two containing the features and targets of the dataset in that order.
     */
    public static List<String[][]> loadFeaturesAndTargets(String filePath, int[] featureColumns, int[] targetColumns) {
        List<String[][]> data = new ArrayList<>();
        String[][] content = FileManager.readCSVtoString(filePath);

        if(featureColumns.length + targetColumns.length != content[0].length) {
            throw new IllegalArgumentException("Total number of passed feature and target columns must match "
            + "the number of columns in the file. However, got (" + featureColumns.length + ", " + targetColumns.length +
                    ") and total columns in the csv file " + content[0].length);
        }

        String[][] features = new String[content.length][featureColumns.length];
        String[][] targets = new String[content.length][targetColumns.length];

        int colCount = 0;
        for(int j : featureColumns) {
            for(int i=0; i< content.length; i++) {
                features[i][colCount] = content[i][j];
            }
            colCount++;
        }

        colCount=0;
        for(int j : targetColumns) {
            for(int i=0; i< content.length; i++) {
                targets[i][colCount] = content[i][j];
            }
            colCount++;
        }

        data.add(features);
        data.add(targets);

        return data;
    }
}
