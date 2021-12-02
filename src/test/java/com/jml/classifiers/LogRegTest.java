package com.jml.classifiers;

import com.jml.core.DataLoader;
import com.jml.core.Model;
import com.jml.core.Stats;
import com.jml.optimizers.Scheduler;
import com.jml.optimizers.StepLearningRate;
import com.jml.preprocessing.Normalize;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogRegTest {

    LogisticRegression logReg;
    List<String[][]> data;
    String filePath;
    double[][] features;
    double[] targets;

    @BeforeEach
    void setUp() {
        filePath = "./src/test/java/com/jml/classifiers/testfiles/winequality-red.csv";
        logReg = new LogisticRegression(0.001);
        data = DataLoader.loadFeaturesAndTargets(filePath);
        features = ArrayUtils.toDouble(data.get(0));
        double[][] quality = ArrayUtils.toDouble(data.get(1));
        targets = processTargets(quality);
    }


    @Test
    // TODO: Add test for prediction method.
    void logRegWineTest() {
        List<String[][]> data = DataLoader.loadFeaturesAndTargets(filePath);

        double learningRate = 0.001;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        // Training data
        double[][] features = ArrayUtils.toDouble(data.get(0));
        features = Normalize.zScore(features); // Normalize the features
        double[][] quality = ArrayUtils.toDouble(data.get(1));
        double[] targets = processTargets(quality);

        // Train model
        logReg.fit(features, targets);
        double[] lossHist = logReg.getLossHist();

        logReg.saveModel("./src/test/java/com/jml/classifiers/testfiles/testLogReg.mdl");

        Model<double[][], double[]> loadedMdl = Model.load("./src/test/java/com/jml/classifiers/testfiles/testLogReg.mdl");

        assertEquals(learningRate, logReg.learningRate);
        assertEquals(threshold, logReg.threshold);
        assertEquals(maxIterations, logReg.maxIterations);
        assertNull(logReg.schedule);
        assertTrue(logReg.isFit);
        assertEquals(loadedMdl.getDetails(), logReg.getDetails());
    }


    @Test
    void logRegTest() {
        double learningRate = 0.2;
        double threshold = 0.5e-5;
        int maxIterations = 1000;

        double[][] features = {{-3}, {-1}, {3.1}, {4}, {5}, {5.12}, {7}, {10}, {12}};
        double[] targets = {0, 0, 0, 0, 1, 1, 1, 1, 1};
        double[][] tests = {{1}, {-10}, {4}, {5}, {6.12}, {100}};
        double[] expected = {0, 0, 0, 1, 1, 1};


        logReg = new LogisticRegression(0.2);
        logReg.fit(features, targets);
        double[] pred = logReg.predict(tests);

        assertEquals(learningRate, logReg.learningRate);
        assertEquals(threshold, logReg.threshold);
        assertEquals(maxIterations, logReg.maxIterations);
        assertArrayEquals(expected, round(pred));
        assertEquals(logReg.w, logReg.getParams());
    }


    @Test
    void logRegConstructorTest() {
        logReg = new LogisticRegression();
        assertEquals(0.002, logReg.learningRate);
        assertEquals(0.5e-5, logReg.threshold);
        assertEquals(1000, logReg.maxIterations);

        logReg = new LogisticRegression(0.1, 1500);
        assertEquals(0.1, logReg.learningRate);
        assertEquals(0.5e-5, logReg.threshold);
        assertEquals(1500, logReg.maxIterations);


        logReg = new LogisticRegression(0.002, 1500, 0.2E-2);
        assertEquals(0.002, logReg.learningRate);
        assertEquals(0.2E-2, logReg.threshold);
        assertEquals(1500, logReg.maxIterations);

        Scheduler schedule = new StepLearningRate(0.8, 2);
        logReg = new LogisticRegression(0.012, 1600, 0.004, schedule);
        assertEquals(0.012, logReg.learningRate);
        assertEquals(0.004, logReg.threshold);
        assertEquals(1600, logReg.maxIterations);
        assertEquals(schedule, logReg.schedule);
        assertEquals(logReg.toString(), logReg.getDetails());
    }


    @Test
    void logRegExceptionTests() {
        logReg = new LogisticRegression();
        double[][] features = {{-3}, {-1}, {3.1}, {4}, {5}, {5.12}, {7}, {10}, {12}};
        double[] targets = {0, 0, 0, 0, 1, 1, 1, 1};

        assertThrows(IllegalArgumentException.class, () -> logReg.fit(features, targets));
        assertThrows(IllegalStateException.class, () -> logReg.predict(features));
        assertThrows(IllegalStateException.class, () -> logReg.getParams());
        assertThrows(IllegalStateException.class, () -> logReg.getLossHist());
        assertThrows(IllegalStateException.class, () -> logReg.saveModel("model.mdl"));

        logReg.fit(new double[][]{{1}}, new double[]{1});

        assertThrows(IllegalArgumentException.class, () -> logReg.saveModel("model.txt"));
    }


    // Quality ratings are on a scale from 1-10. We wil say a wine is good quality if its rating is >= 7
    static double[] processTargets(double[][] targets) {
        double[] result = new double[targets.length];

        for(int i=0; i<targets.length; i++) {
            if(targets[i][0] >= 7) {
                result[i] = 1; // Good Quality
            } else {
                result[i] = 0; // Bad Quality
            }
        }

        return result;
    }


    static double[] round(double[] data) {
        double[] result = new double[data.length];

        for(int i=0; i<data.length; i++) {
            result[i] = Stats.round(data[i], 0);
        }

        return result;
    }
}
