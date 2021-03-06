package com.jml.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class OneHotEncoderTest {
    OneHotEncoder enc;

    @BeforeEach
    void setup() {
        enc = new OneHotEncoder(true);
    }


    @Test
    void singleDimensionTest () {
        String[][] data = {{"male"}, {"female"}, {"female"}, {"dog"}, {"male"}};
        int[][] expDataEncodings = {{0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

        String[][] val = {{"female"}, {"male"}, {"child"}, {"male"}, {"cat"}, {"dog"}, {"female"}};
        String[][] expValDecodings = {{"female"}, {"male"}, null, {"male"}, null, {"dog"}, {"female"}};
        int[][] expValEncodings = {{0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}};

        enc.fit(data);

        int[][] dataEncodings = enc.encode(data);
        int[][] valEncodings = enc.encode(val);

        String[][] invDataEncodings = enc.decode(expDataEncodings);
        String[][] invValEncodings = enc.decode(expValEncodings);

        assertArrayEquals(expDataEncodings, dataEncodings);
        assertArrayEquals(expValEncodings, valEncodings);
        assertArrayEquals(data, invDataEncodings);
        assertArrayEquals(expValDecodings, invValEncodings);
    }


    @Test
    void twoDimensionTest() {
        String[][] data = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}};
        int[][] expDataEncodings = {{0, 0, 1, 1, 0},
                                    {0, 1, 0, 1, 0},
                                    {0, 1, 0, 0, 1},
                                    {1, 0, 0, 0, 1},
                                    {0, 0, 1, 0, 1}};

        String[][] val = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}, {"cat", "1"}};
        String[][] expValDecodings = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}, null};
        int[][] expValEncodings = { {0, 0, 1, 1, 0},
                                    {0, 1, 0, 1, 0},
                                    {0, 1, 0, 0, 1},
                                    {1, 0, 0, 0, 1},
                                    {0, 0, 1, 0, 1},
                                    {0, 0, 0, 0, 0}};


        enc.fit(data);

        int[][] dataEncodings = enc.encode(data);
        int[][] valEncodings = enc.encode(val);

        String[][] invDataEncodings = enc.decode(expDataEncodings);
        String[][] invValEncodings = enc.decode(expValEncodings);

        assertArrayEquals(expDataEncodings, dataEncodings);
        assertArrayEquals(expValEncodings, valEncodings);
        assertArrayEquals(data, invDataEncodings);
        assertArrayEquals(expValDecodings, invValEncodings);
    }


    @Test
    void twoDimensionNoIgnoreTest() {
        enc = new OneHotEncoder();
        String[][] data = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}};
        int[][] expDataEncodings = {{0, 0, 1, 1, 0},
                {0, 1, 0, 1, 0},
                {0, 1, 0, 0, 1},
                {1, 0, 0, 0, 1},
                {0, 0, 1, 0, 1}};

        String[][] val = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}};
        String[][] expValDecodings = {{"male", "0"}, {"female", "0"}, {"female", "1"}, {"dog", "1"}, {"male", "1"}};
        int[][] expValEncodings = { {0, 0, 1, 1, 0},
                {0, 1, 0, 1, 0},
                {0, 1, 0, 0, 1},
                {1, 0, 0, 0, 1},
                {0, 0, 1, 0, 1}};


        enc.fit(data);

        int[][] dataEncodings = enc.encode(data);
        int[][] valEncodings = enc.encode(val);

        String[][] invDataEncodings = enc.decode(expDataEncodings);
        String[][] invValEncodings = enc.decode(expValEncodings);

        assertArrayEquals(expDataEncodings, dataEncodings);
        assertArrayEquals(expValEncodings, valEncodings);
        assertArrayEquals(data, invDataEncodings);
        assertArrayEquals(expValDecodings, invValEncodings);


        assertThrows(Exception.class, () -> enc.encode(new String[][]{{"cat", "1"}}));
    }
}
