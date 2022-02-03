package com.jml.preprocessing;

import com.jml.core.DataLoader;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DataSplitterTest {
    final String filePath = "./src/test/java/com/jml/preprocessing/data/testdata.csv";

    @Test
    void trainTestSplitTest() {
        int[] x_cols = {0, 1, 2};
        int[] y_cols = {3};
        List<String[][]> data = DataLoader.loadFeaturesAndTargets(filePath, x_cols, y_cols);

        double[][] x = ArrayUtils.toDouble(data.get(0));
        double[][] y = ArrayUtils.toDouble(data.get(1));

        Map<String, double[][]> split = DataSplitter.trainTestSplit(x, y, 0.4);
        double[][] x_train = split.get("xTrain");
        double[][] y_train = split.get("yTrain");
        double[][] x_test = split.get("xTest");
        double[][] y_test = split.get("yTest");

        assertEquals(x_train.length, y_train.length);
        assertEquals(x_test.length, y_test.length);
        assertEquals(6, x_train.length);
        assertEquals(4, x_test.length);
    }
}
