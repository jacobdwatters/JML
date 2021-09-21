package com.jml.util;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class FileManager {

    private FileManager() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }

    public static void stringToFile(String data, String filePath) {

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(filePath));
            writer.write(data);
            writer.close();
        } catch (IOException e) {
            System.err.print("Could not write to file " + filePath);
        }
    }

}
