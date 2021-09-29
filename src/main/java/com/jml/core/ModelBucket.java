package com.jml.core;

import java.util.HashMap;
import java.util.Map;


/**
 * A container for holding the results of a fitted model. This Object will hold different
 * data depending on the model.<br><br>
 *
 * All data is accessed through the various <code>get*(String)</code> methods:
 * <pre>
 *     - {@link #getDoubleArr(String)}
 *     - {@link #getDoubleArr2D(String)}
 *     - {@link #getDouble(String)}
 *     - {@link #getInteger(String)}
 * </pre>
 *
 * If you need to get the Class of the object in the ModelBucket, you can use {@link #type(String)}
 */
public class ModelBucket {
    Map<String, Object> bucket = new HashMap<>();


    /**
     * Creates a model bucket from the elements of a map.
     * @param bucketElements Elements to copy to bucket.
     */
    public ModelBucket(Map<String, Object> bucketElements) {
        for (Map.Entry<String, Object> entry : bucketElements.entrySet()) {
            bucket.put(entry.getKey(), entry.getValue());
        }
    }


    /**
     * Checks if a key exists in this bucket and that the data associated with this key is not null.
     *
     * @param key Key associated with the data of interests.
     * @return
     */
    public boolean containsKeyNotNull(String key) {
        return (bucket.containsKey(key) && bucket.get(key) != null);
    }


    /**
     * Gets the class type of the object in this bucket associated with the given key.<br><br>
     *
     * It is recommended to call {@link #containsKeyNotNull(String)} before this method to ensure the key exists in this
     * bucket. If the key does not exist, or if it exists and is null, this method will throw a null pointer
     * exception.
     *
     * @param key Key associated with the data of interests.
     * @return The Class of the object in this bucket which
     * @throws NullPointerException If the object does not exist or if it exists and is null.
     */
    public Class type(String key) {
        if(containsKeyNotNull(key)) {
            return bucket.get(key).getClass();
        } else {
            throw new NullPointerException("Object exists but is null. Can not get type of null.");
        }
    }


    /**
     * Looks up the double array associated with the given key.
     *
     * @param key Key associated double array.
     * @return Returns the double array associated with the provided key if it exists and has the correct type.
     * If there is no value for that key, then the method will return null.
     */
    public double[] getDoubleArr(String key) {
        Object data = null;
        double[] dataAsArr = null;

        if(bucket.containsKey(key)) {
            data = bucket.get(key);

            if(data instanceof double[]) {
                dataAsArr = (double[]) data;
            }
        }

        return dataAsArr;
    }


    /**
     * Looks up the 2D double array associated with the given key.
     *
     * @param key Key associated 2D double array.
     * @return Returns the 2D double array associated with the provided key if it exists and has the correct type.
     * If there is no value for that key, then the method will return null.
     */
    public double[][] getDoubleArr2D(String key) {
        Object data = null;
        double[][] dataAsArr = null;

        if(bucket.containsKey(key)) {
            data = bucket.get(key);

            if(data instanceof double[][]) {
                dataAsArr = (double[][]) data;
            }
        }

        return dataAsArr;
    }


    /**
     * Looks up the Double associated with the given key.
     *
     * @param key Key associated Double.
     * @return Returns the Double associated with the provided key if it exists and has the correct type.
     * If there is no value for that key, then the method will return null.
     */
    public Double getDouble(String key) {
        Object data = null;
        Double dataAsArr = null;

        if(bucket.containsKey(key)) {
            data = bucket.get(key);

            if(data instanceof Double) {
                dataAsArr = (Double) data;
            }
        }

        return dataAsArr;
    }


    /**
     * Looks up the Integer associated with the given key.
     *
     * @param key Key associated Integer.
     * @return Returns the Integer associated with the provided key if it exists and has the correct type.
     * If there is no value for that key, then the method will return null.
     */
    public Object getInteger(String key) {
        Object data = null;
        Integer dataAsArr = null;

        if(bucket.containsKey(key)) {
            data = bucket.get(key);

            if(data instanceof Integer) {
                dataAsArr = (Integer) data;
            }
        }

        return dataAsArr;
    }
}
