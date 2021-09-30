package com.jml.util;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;


public class FileManager {
    private FileManager() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * Writes a string to a file.
     * @param data Raw data to write to file.
     * @param filePath File path including file extension/
     */
    public static void stringToFile(String data, String filePath) {

        try(BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write(data);
        } catch (IOException e) {
            System.err.print("Could not write to file " + filePath);
        }
    }


    /**
     * Reads content from a file.
     * @return Raw content of file.
     */
    public static String readFile(String filePath) {
        StringBuilder contentBuilder = new StringBuilder();

        try (Stream<String> stream = Files.lines(Paths.get(filePath), StandardCharsets.UTF_8)) {
            stream.forEach(s -> contentBuilder.append(s).append("\n"));
        }
        catch (IOException e) {
            System.err.println("Could not read file " + filePath);
        }

        return contentBuilder.toString();
    }
}
