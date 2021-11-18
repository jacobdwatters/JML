package com.jml.classifiers;

import com.jml.core.DataLoader;
import com.jml.core.Model;
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


//    @Test
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

    // Average difference of neighboring elements.
    protected static double aveSlope(double[] arr) {
        return (arr[0]-arr[arr.length-1])/2;
    }
}
