package linalg.util;

import linalg.Matrix;
import linalg.complex_number.CNumber;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * This class provides several methods for loading and saving matrices to/from a file.
 * 
 * @author Jacob Watters
 */
public class FileManager {
	
	/**
	 * Loads matrix data from a file.
	 * 
	 * @throws IllegalArgumentException if file path does not end with .csv, if file data is not formatted to be a CNumber correctly.
	 * @param filePath - The path, including extension, of the file to read.
	 * @return A matrix containing the data from the csv file.
	 */
	public static Matrix readcsv(String filePath) {
		if(!filePath.substring(filePath.length()-4, filePath.length()).equals(".csv")) {
			throw new IllegalArgumentException("Ensure file ends with .csv");
		}
		
		File file= new File(filePath);

        // this gives you a 2-dimensional array of strings
        List<List<String>> lines = new ArrayList<>();
        Scanner inputStream;

        try{
            inputStream = new Scanner(file);

            while(inputStream.hasNext()){
                String line = inputStream.nextLine();
                String[] values = line.split(",");
                // this adds the currently parsed line to the 2-dimensional string array
                lines.add(Arrays.asList(values));
            }

            inputStream.close();
        }catch (FileNotFoundException e) {
            System.err.println("Could not find file " + filePath);
        }

        System.out.println("m: " + lines.size());
        System.out.println("n: " + lines.get(0).size());
        
        CNumber[][] arr = new CNumber[lines.size()][lines.get(0).size()];
        // the following code lets you iterate through the 2-dimensional array
        int lineNo = 0;
        for(List<String> line: lines) {
            int columnNo = 0;
            
            for (String value: line) {
                arr[lineNo][columnNo] = new CNumber(value);
                columnNo++;
            }
            lineNo++;
        }
		
		return new Matrix(arr);
	}
	
	
	/**
	 * Writes the contends of a matrix to a csv file.
	 * 
	 * @param A - Matrix to write to file.
	 * @param filePath - Path, including extension, to write file to.
	 */
	public static void write2csv(Matrix A, String filePath) {
		if(!filePath.substring(filePath.length()-4, filePath.length()).equals(".csv")) {
			throw new IllegalArgumentException("Ensure file ends with .csv");
		}
		
		String contents = "";
		
		for(int i=0; i<A.numRows(); i++) {
			for(int j=0; j<A.numCols(); j++) {
				contents += A.get(i, j);
				
				if(j!=A.numCols()-1) {
					contents += ", ";
				}
			}
			
			contents += "\n";
		}
		
		try (PrintWriter out = new PrintWriter(filePath)) {
		    out.println(contents);
		} catch (FileNotFoundException e) {
			System.err.println("Could not find file " + filePath);
		}
	}
	
	
	public static void main(String[] args) {
		
		String[][] a = {{"3-21i", "i", "1-i"}, 
						{"5", "123-1.2i", "2i"}};
		Matrix A = new Matrix(a);
		
		write2csv(A, "C:\\Users\\17194\\Desktop\\test.csv");
		Matrix.print(readcsv("C:\\Users\\17194\\Desktop\\test.csv"));
	}
}
