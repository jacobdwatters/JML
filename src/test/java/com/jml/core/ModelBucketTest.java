package com.jml.core;

import org.junit.jupiter.api.Test;
import java.util.HashMap;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ModelBucketTest {
    ModelBucket bucket;

    @Test
    void ModelDoubleArrBucketTest() {
        double[] arr1 = {1, 2, 3, 4};
        double[] arr2 = {5};

        Map<String, Object> map = new HashMap<>();
        map.put("1", arr1);
        map.put("2", arr2);

        bucket = new ModelBucket(map);

        double[] arr1Actual = bucket.getDoubleArr("1");
        double[] arr2Actual = bucket.getDoubleArr("2");

        assertArrayEquals(arr1, arr1Actual);
        assertArrayEquals(arr2, arr2Actual);
    }


    @Test
    void ModelDoubleArr2DBucketTest() {
        double[][] arr1 = {{1, 2, 3, 4},
                            {5, 6, 7, 8}};
        double[][] arr2 = {{5}};

        Map<String, Object> map = new HashMap<>();
        map.put("1", arr1);
        map.put("2", arr2);

        bucket = new ModelBucket(map);

        double[][] arr1Actual = bucket.getDoubleArr2D("1");
        double[][] arr2Actual = bucket.getDoubleArr2D("2");

        assertArrayEquals(arr1, arr1Actual);
        assertArrayEquals(arr2, arr2Actual);
    }


    @Test
    void ModelDoubleBucketTest() {
        double value = -0.23423;

        Map<String, Object> map = new HashMap<>();
        map.put("1", value);

        bucket = new ModelBucket(map);

        double value1Actual = bucket.getDouble("1");

        assertEquals(value, value1Actual);
    }


    @Test
    void ModelIntegerBucketTest() {
        int value = 5;

        Map<String, Object> map = new HashMap<>();
        map.put("1", value);

        bucket = new ModelBucket(map);

        int value1Actual = bucket.getInteger("1");

        assertEquals(value, value1Actual);
    }


    @Test
    void ModelNullTest() {
        int value = 5;

        Map<String, Object> map = new HashMap<>();
        map.put("1", value);

        bucket = new ModelBucket(map);
        Integer value1Actual = bucket.getInteger("4");

        assertNull(value1Actual);
    }


    @Test
    void ModelBucketType() {
        Integer value = 5;

        Map<String, Object> map = new HashMap<>();
        map.put("1", value);

        bucket = new ModelBucket(map);

        assertEquals(value.getClass(), bucket.type("1"));
    }
}
