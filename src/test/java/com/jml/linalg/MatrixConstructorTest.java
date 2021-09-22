package com.jml.linalg;

import linalg.Matrix;
import linalg.complex_number.CNumber;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixConstructorTest {
    Matrix m;

    @Test // Defines a test method
    @DisplayName("Checking Default matrix constructor") // define the name of the test which is displayed to the user
    void defaultConstructorTestCase() {
        m = new Matrix();
        CNumber[][] expected = new CNumber[0][0];

        assertEquals(m.numCols(), 0);
        assertEquals(m.numRows(), 0);
        assertEquals(m.shape(), "0x0");
        assertArrayEquals(expected, m.getValues());
    }

    @Test // Defines a test method
    @DisplayName("Checking size matrix constructor") // define the name of the test which is displayed to the user
    void sizeConstructorTestCase() {
        for(int size1 = 0; size1<15; size1++) {
            m = new Matrix(size1);
            CNumber[][] expected = new CNumber[size1][size1];
            for (int i = 0; i < expected.length; i++) {
                for (int j = 0; j < expected[0].length; j++)
                    expected[i][j] = CNumber.ZERO;
            }

            assertEquals(m.numCols(), size1);
            assertEquals(m.numRows(), size1);
            assertEquals(m.shape(), size1 + "x" + size1);
            assertArrayEquals(expected, m.getValues());
        }
    }

}
