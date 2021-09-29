package com.jml.core;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;

class ModelBucketExceptionTest {

    @Test // Defines a test method
    @DisplayName("Test for attempted fitting/prediction but model is not compiled yet.") // define the name of the test which is displayed to the user
    void notCompiledTestCase() {
        Map<String, Object> map = new HashMap<>();

        Integer i = null;
        map.put("1", i);

        ModelBucket bucket = new ModelBucket(map);

        assertThrows(Exception.class, () -> bucket.type("1"));
        assertThrows(Exception.class, () -> bucket.type("3"));
    }
}
