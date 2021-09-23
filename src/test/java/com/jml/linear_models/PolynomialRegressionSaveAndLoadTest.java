package com.jml.linear_models;

import com.jml.core.Model;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class PolynomialRegressionSaveAndLoadTest {

    Model<double[], double[]> model;
    Model<double[], double[]> loadedModel;
    double[] features;
    double[] targets;
    double[] tests;

    @BeforeEach
    void setUp() {
        model = new PolynomialRegression();
        loadedModel = new PolynomialRegression();
        features = new double[]{1, 4, 5, 6, 7};
        targets = new double[]{3, 4, 5, 6, 10};
        tests = new double[]{-1.2, 0, 1, 5};
    }

    @Test // Defines a test method
    @DisplayName("Checks if model is saved and loaded properly.") // define the name of the test which is displayed to the user
    void SaveAndLoadModelDegree1TestCase() {
        String filePath = "./src/test/java/com/jml/linear_models/test_model_files/testPolyModel1.mdl";

        model.compile();
        double[][] c = model.fit(features, targets);
        double[] initialPredictions = model.predict(tests);
        model.saveModel(filePath);

        loadedModel = Model.load(filePath);

        assertEquals(model.getDetails(), loadedModel.getDetails());
        assertArrayEquals(initialPredictions, loadedModel.predict(tests));
    }


    @Test // Defines a test method
    @DisplayName("Checks if model is saved and loaded properly.") // define the name of the test which is displayed to the user
    void SaveAndLoadModelDegree3TestCase() {
        String filePath = "./src/test/java/com/jml/linear_models/test_model_files/testPolyModel2.mdl";

        Map<String, Double> args = new HashMap<>();
        args.put(PolynomialRegression.DEGREE_KEY, 3.0);
        args.put(PolynomialRegression.NORMALIZE_KEY, 1.0);

        model.compile(args);
        double[][] c = model.fit(features, targets);
        double[] initialPredictions = model.predict(tests);
        model.saveModel(filePath);

        loadedModel = Model.load(filePath);

        assertEquals(model.getDetails(), loadedModel.getDetails());
        assertArrayEquals(initialPredictions, loadedModel.predict(tests));
    }


    @Test // Defines a test method
    @DisplayName("Checks if exception is thrown if model is saved before being fit") // define the name of the test which is displayed to the user
    void SaveAndModelNotFitExceptionTestCase() {
        String filePath = "./src/test/java/com/jml/linear_models/test_model_files/testPolyModel3.mdl";

        Map<String, Double> args = new HashMap<>();
        args.put(PolynomialRegression.DEGREE_KEY, 10.0);

        model.compile(args);
        assertThrows(Exception.class, () -> model.saveModel(filePath));
    }


    @Test // Defines a test method
    @DisplayName("Checks if exception is thrown if model has incorrect file extension.") // define the name of the test which is displayed to the user
    void SaveAndModelIncorrectFileExceptionTestCase() {
        String filePath = "./src/test/java/com/jml/linear_models/test_model_files/testPolyModel3.txt";

        Map<String, Double> args = new HashMap<>();
        args.put(PolynomialRegression.DEGREE_KEY, 2.0);

        model.compile(args);
        double[][] c = model.fit(features, targets);
        assertThrows(Exception.class, () -> model.saveModel(filePath));
        assertThrows(Exception.class, () -> Model.load(filePath));
    }
}
