package com.jml.classifiers;

import com.jml.core.DataLoader;
import com.jml.core.Model;
import com.jml.preprocessing.Encoder;
import com.jml.util.ArrayUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

class KnnTest {

    Model<double[][], int[]> knn;
    List<String[][]> data;
    String filePath;
    double[][] features;
    int[] classes;

    @BeforeEach
    void setUp() {
        filePath = "./src/test/java/com/jml/classifiers/testfiles/test.csv";

        data = DataLoader.loadFeaturesAndTargets(filePath);
        features = ArrayUtils.toDouble(data.get(0));
        String[][] targets = data.get(1);
        classes = Encoder.encodeClasses(targets);
    }


    @Test
    void Knnk1TestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors(1);
        knn.fit(features, classes);

        double[][] tests = {{7.6, 7.1}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk2TestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors();
        knn.fit(features, classes);

        double[][] tests = {{7.6, 7.1}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk3TestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors(3);
        knn.fit(features, classes);

        double[][] tests = {{7.6, 7.1}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk4TestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors(4);
        knn.fit(features, classes);

        double[][] tests = {{7.6, 7.1}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk7TestCase() {
        int[] expectedPred = {1};

        KNearestNeighbors knn = new KNearestNeighbors(7);
        knn.fit(features, classes);

        double[][] tests = {{7.6, 7.1}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk8TestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors(8);
        knn.fit(features, classes);

        double[][] tests = {{7.6, 7.1}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk2MTestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors(2, 1);
        knn.fit(features, classes);

        double[][] tests = {{3.5, 3}};
        int[] predictions = knn.predict(tests);

        String expectedDetails = "Model Details\n" +
                "----------------------------\n" +
                "Model Type: " +  "K_NEAREST_NEIGHBORS" + "\n" +
                "Is Trained: " + "Yes" + "\n" +
                "k-neighbors: " + 2 + "\n" +
                "distance parameter: " + 1;

        assertArrayEquals(expectedPred, predictions);
        assertEquals(expectedDetails, knn.inspect());
    }


    @Test
    void Knnk5MTestCase() {
        int[] expectedPred = {1};

        KNearestNeighbors knn = new KNearestNeighbors(5, 1);
        knn.fit(features, classes);

        double[][] tests = {{3.5, 3}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk6MTestCase() {
        int[] expectedPred = {0};

        KNearestNeighbors knn = new KNearestNeighbors(6, 1);
        knn.fit(features, classes);

        double[][] tests = {{3.5, 3}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void Knnk9MTestCase() {
        int[] expectedPred = {1};

        KNearestNeighbors knn = new KNearestNeighbors(9, 1);
        knn.fit(features, classes);

        double[][] tests = {{3.5, 3}};
        int[] predictions = knn.predict(tests);

        assertArrayEquals(expectedPred, predictions);
    }


    @Test
    void KnnFitExceptionTestCase() {
        int[] expectedPred = {1};

        double[][] feat = new double[5][2];
        int[] claz = new int[3];

        KNearestNeighbors knn = new KNearestNeighbors(9, 1);

        assertThrows(Exception.class, () -> knn.fit(feat, claz));
    }


    @Test
    void knnSaveLoadTest() {
        KNearestNeighbors knn = new KNearestNeighbors(9, 1);
        knn.fit(features, classes);
        knn.saveModel("./src/test/java/com/jml/classifiers/testfiles/testKnn.mdl");
        Model loadedKnn = Model.load("./src/test/java/com/jml/classifiers/testfiles/testKnn.mdl");

        assertEquals(loadedKnn.inspect(), knn.inspect());
    }


    @Test
    void knnSaveLoadExceptionTest() {
        KNearestNeighbors knn = new KNearestNeighbors(9, 1);
        knn.fit(features, classes);

        assertThrows(Exception.class, () -> knn.saveModel("./src/test/java/com/jml/classifiers/testfiles/testKnn"));
    }


    @Test
    void knnSaveLoadException2Test() {
        KNearestNeighbors knn = new KNearestNeighbors(9, 1);

        assertThrows(Exception.class, () -> knn.saveModel("./src/test/java/com/jml/classifiers/testfiles/testKnn"));
    }


    @Test
    void knnPredictException1Test() {
        KNearestNeighbors knn = new KNearestNeighbors(9, 1);
        double[][] tests = {{3.5, 3}};

        assertThrows(Exception.class, () -> knn.predict(tests));
    }
}
