package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;


class MultipleRegressionDatasetOneTest {
    Model<double[][], double[]> model;

    double[][] features;
    double[] targets;
    double[][] tests;
    double[] expectedCoefficients;
    double[] expectedPredictions;


    @BeforeEach // Runs before each test
    void setUp() {
        model = new MultipleLinearRegression();
        features = new double[][] { {1, 4, -3, 4},
                                    {5, 6, -4, 8},
                                    {9, -1, 11, 12},
                                    {0, 1, 0.6, 1},
                                    {5, 4, 3, 2}};
        targets = new double[]{1, 9, -1, 6, 4};
        tests = new double[][] {{1, 3, 4, 5},
                                {-0.1, 3.5, 2, 6}};

        expectedCoefficients = new double[]{20.0672998, 7.5041288, -7.7770438, -5.4995871, -2.9905037};
        expectedPredictions = new double[]{-32.7105698, -36.8449628};
    }

    @Test
    void MultipleRegressionDatasetOneTestCase() {
        model.compile();
        ModelBucket results = model.fit(features, targets);

        double[] actualCoefficients = results.getDoubleArr("coefficients");
        double[] actualPredictions = model.predict(tests);

        assertArrayEquals(expectedCoefficients, ArrayUtils.round(actualCoefficients, 7));
        assertArrayEquals(expectedPredictions, ArrayUtils.round(actualPredictions, 7));
    }


    @Test
    void MultipleRegressionDatasetOneSaveLoadTestCase() {
        model.compile();
        ModelBucket results = model.fit(features, targets);
        String fileName = "./src/test/java/com/jml/linear_models/test_model_files/testMultModel1.mdl";

        String expectedDetails = model.getDetails();
        model.saveModel(fileName);
        Model loadedModel = Model.load(fileName);
        String actualDetails = loadedModel.getDetails();

        assertEquals(expectedDetails, actualDetails);
    }


    @Test
    void MultipleRegressionSaveLoadExceptionsTestCase() {
        model.compile();
        ModelBucket results = model.fit(features, targets);
        String fileName1 = "./src/test/java/com/jml/linear_models/test_model_files/testMultModel1.txt";
        String fileName2 = "./src/test/java/com/jml/linear_models/test_model_files/testMultModel0.mdl";

        assertThrows(Exception.class, () -> model.saveModel(fileName1));
        assertThrows(Exception.class, () -> Model.load(fileName1));
        assertNull(Model.load(fileName2));
    }


    @Test
    void MultipleRegressionNotCompiledTestCase() {
        String fileName = "./src/test/java/com/jml/linear_models/test_model_files/testMultModel0.mdl";

        assertThrows(Exception.class, () -> model.fit(features, targets));
        assertThrows(Exception.class, () -> model.saveModel(fileName));
        assertThrows(Exception.class, () -> model.predict(tests));
    }


    @Test
    void MultipleRegressionNotFitTestCase() {
        String fileName = "./src/test/java/com/jml/linear_models/test_model_files/testMultModel0.mdl";
        model.compile();

        assertThrows(Exception.class, () -> model.saveModel(fileName));
        assertThrows(Exception.class, () -> model.predict(tests));
    }
}
