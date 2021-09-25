package com.jml.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class ArrayErrorsTest {

    double[] dataset1, dataset2, dataset3;


    @BeforeEach // Runs before each test
    void setUp() {
        dataset1 = new double[]{};
        dataset2 = new double[]{1, 2, 3, 4};
        dataset3 = new double[]{1, 2, 3};
    }

    @Test // Defines a test method
    @DisplayName("Test for empty array error") // define the name of the test which is displayed to the user
    void emptyArrayExceptionTestCase() {
        assertThrows(Exception.class, () -> ArrayErrors.checkNotEmpty(dataset1));
    }

    @Test // Defines a test method
    @DisplayName("Test for different length arrays error") // define the name of the test which is displayed to the user
    void differentLengthArraysExceptionTestCase() {
        assertThrows(Exception.class, () -> ArrayErrors.checkSameLength(dataset2, dataset3));
    }
}
