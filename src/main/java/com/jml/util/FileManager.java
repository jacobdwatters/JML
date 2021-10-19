package com.jml.util;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;


public class FileManager {
    private FileManager() {
        throw new IllegalStateException("Utility class, Can not create instantiated.");
    }


    /**
     * Writes a string to a file.
     * @param data Raw data to write to file.
     * @param filePath File path including file extension
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
     * @param filePath File path including file extension of the file to load.
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


    /**
     * Reads a CSV (comma seperated value) file into a 2D array of String.
     *
     * @param filePath File name including the extension. The extension must be .csv
     * @return The CSV file contents as a 2D array of Strings
     */
    public static String[][] readCSVtoString(String filePath) {
        Scanner sc;
        String[] arr;
        String[][] csv= null;

        try {
            sc = new Scanner(new File(filePath));
            List<String> lines = new ArrayList<>();
            while (sc.hasNextLine()) {
                lines.add(sc.nextLine());
            }

            arr = lines.toArray(new String[0]);
            csv = new String[arr.length][arr[0].split(",").length];

            for(int i = 0; i < arr.length; i++) {
                csv[i] = arr[i].split(",");
            }


        } catch (FileNotFoundException e) {
            System.err.println("Error: Must pass file names as commandline arguments or provide a list of file names in FileList.txt");
            e.printStackTrace();
        }

        return csv;
    }


    /**
     * Reads a CSV (comma seperated value) file into a 2D array of doubles.
     *
     * @param filePath File name including the extension. The extension must be .csv
     * @return The CSV file contents as a 2D array of doubles
     */
    public static double[][] readCSVtoDouble(String filePath) {
        String[][] fileContent = readCSVtoString(filePath);
        return ArrayUtils.toDouble(fileContent);
    }
}
