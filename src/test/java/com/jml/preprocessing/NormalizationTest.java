package com.jml.preprocessing;

import com.jml.core.Stats;
import com.jml.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class NormalizationTest {

    double[] dataOne;
    double[] dataTwo;
    double[][] data2D;
    double[] expectedOne;
    double[] expectedTwo;
    double[][] expected2D;

    @BeforeEach
    void setUp() {
        dataOne = new double[]{1, 2, 3, 4, 5};
        dataTwo = new double[]{0.12, 12.2342, -123.92342, 193.9244};
        data2D = new double[][]{{1, 2, 3, 4, 5},
                                {2, 4, 6, 1, 7},
                                {4, 65, 2, 6, 8}};
    }

    @Test
    void l2NormalizeTestCase() {
        expectedOne = new double[]{0.13483997, 0.26967994, 0.40451992, 0.53935989, 0.67419986};
        expectedTwo = new double[]{0.00052069, 0.05308522, -0.53771411, 0.84145423};
        expected2D = new double[][]
                {{0.13483997, 0.26967994, 0.40451992, 0.53935989, 0.67419986},
                {0.19425717, 0.38851434, 0.58277152, 0.09712859, 0.6799001, },
                {0.06068273, 0.98609434, 0.03034136, 0.09102409, 0.12136546}};

        assertArrayEquals(expectedOne, ArrayUtils.round(Normalize.l2(dataOne), 8));
        assertArrayEquals(expectedTwo, ArrayUtils.round(Normalize.l2(dataTwo), 8));
        assertArrayEquals(expected2D, ArrayUtils.round(Normalize.l2(data2D), 8));
    }


    @Test
    void meanNormalizeTestCase() {
        expectedOne = new double[]{-0.5, -0.25, 0.0, 0.25, 0.5};
        expectedTwo = new double[]{-0.06439809780667993, -0.026284890045808713, -0.45465850607375574, 0.5453414939262443};

        assertArrayEquals(expectedOne, Normalize.meanNormalize(dataOne));
        assertArrayEquals(expectedTwo, Normalize.meanNormalize(dataTwo));
    }


    @Test
    void minMaxScaleNormalizeTestCase() {
        expectedOne = new double[]{0.0, 0.25, 0.5, 0.75, 1.0};
        expectedTwo = new double[]{0.39026040826707575, 0.42837361602794694, 0.0, 1.0};

        assertArrayEquals(expectedOne, Normalize.minMaxScale(dataOne), 8);
        assertArrayEquals(expectedTwo, Normalize.minMaxScale(dataTwo), 8);
    }


    @Test
    void minMaxABScaleNormalizeTestCase() {
        expectedOne = new double[]{0.0, 1.0, 2.0, 3.0, 4.0};
        expectedTwo = new double[]{0.7805208165341515, 0.8567472320558939, 0.0, 2.0};

        assertArrayEquals(expectedOne, Normalize.minMaxScale(dataOne, 3, 7));
        assertArrayEquals(expectedTwo, Normalize.minMaxScale(dataTwo, -1, 1));
    }


    @Test
    void zScoreNormalizeTestCase() {
        expectedOne = new double[]{-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518};
        expectedTwo = new double[]{-0.1563494039699112, -0.0638159671177517, -1.10384605843293, 1.324011429520593};

        assertArrayEquals(expectedOne, Normalize.zScore(dataOne), 8);
        assertArrayEquals(expectedTwo, Normalize.zScore(dataTwo), 8);
    }
}
