package com.jml.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class DataLoaderTest {

    String filePath;
    String[][] dataset = {{"1", "2", "3", "4", "5"},
            {"10", "20", "30", "40", "50"},
            {"100", "200", "300", "400", "500"},
            {"1000", "2000", "3000", "4000", "5000"},
            {"0.1", "0.2", "0.3", "0.4", "0.5"}};

    @BeforeEach
    void setUp() {
        filePath = "./src/test/java/com/jml/core/testfiles/loadDataTest.csv";
    }

    @Test
    void standardLoadTest() {
        List<String[][]> data = DataLoader.loadFeaturesAndTargets(filePath);

        String[][] expX = {{"1", "2", "3", "4"},
                {"10", "20", "30", "40"},
                {"100", "200", "300", "400"},
                {"1000", "2000", "3000", "4000"},
                {"0.1", "0.2", "0.3", "0.4"}};

        String[][] expy = {{"5"},
                {"50"},
                {"500"},
                {"5000"},
                {"0.5"}};

        String[][] X = data.get(0);
        String[][] y = data.get(1);

        assertEquals(2, data.size());
        assertArrayEquals(expX, X);
        assertArrayEquals(expy, y);
    }

    @Test
    void specifiedLoadTest() {
        int[] F = {0, 2};
        int[] T = {1, 3, 4};
        List<String[][]> data = DataLoader.loadFeaturesAndTargets(filePath, F, T);

        String[][] expX = {{"1", "3"},
                {"10", "30"},
                {"100", "300"},
                {"1000", "3000"},
                {"0.1", "0.3"}};

        String[][] expy = {{"2", "4", "5"},
                {"20", "40", "50"},
                {"200", "400", "500"},
                {"2000", "4000", "5000"},
                {"0.2", "0.4", "0.5"}};

        String[][] X = data.get(0);
        String[][] y = data.get(1);

        assertEquals(2, data.size());
        assertArrayEquals(expX, X);
        assertArrayEquals(expy, y);

        assertThrows(IllegalArgumentException.class, () -> DataLoader.loadFeaturesAndTargets(filePath,
                new int[]{1, 2}, new int[]{3, 4}));
    }
}
