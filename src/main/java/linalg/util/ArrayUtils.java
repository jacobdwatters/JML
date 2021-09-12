package linalg.util;

import linalg.complex_number.CNumber;

public class ArrayUtils {
	
	public static CNumber[][] random(int rows, int cols, double min, double max, boolean... magnitude_flag) {
		// TODO: Cant have more than one flag
		
		boolean flag = false;
		CNumber[][] result = new CNumber[rows][cols];
		
		if(magnitude_flag.length > 0) {
			flag = magnitude_flag[0];
		}
		

		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[0].length; j++) {
				result[i][j] = CNumber.random(min, max, flag);
			}
		}
		
		return result;
	}
	
	
	public static CNumber[][] random(int rows, int cols) {
		return random(rows, cols, 0, 1);
	}
	
	
	public static CNumber[][] random(int rows, int cols, double mag) {
		CNumber[][] result = new CNumber[rows][cols];
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				result[i][j] = CNumber.random(mag);
			}
		}
		
		return result; 
	}
	
	
	public static CNumber[][] randomComplex(int rows, int cols) {
		CNumber[][] values = new CNumber[rows][cols];
		
		for(int i=0; i<rows; i++) {
			for(int j=0; j<cols; j++) {
				values[i][j] = CNumber.randn(true); 
			}
		}
		
		return values;
	}
	
	
	public static CNumber[][] randn(int rows, int cols, boolean complex) {
		CNumber[][] values = new CNumber[rows][cols];
		
		for(int i=0; i<rows; i++) {
			for(int j=0; j<cols; j++) {
				values[i][j] = CNumber.randn(complex); 
			}
		}
		
		return values;
	}
	
	
	public static void printArr(CNumber[][] arr) {
		for(int i = 0; i < arr.length; i++) {
			for(int j = 0; j < arr[0].length; j++) {
				System.out.print(arr[i][j] + ", ");
			}
			System.out.println();
		}
	}
	
	
	public static void printArr(CNumber[] arr) {
		for(int i = 0; i < arr.length; i++) {
			System.out.print(arr[i] + ", ");
		}
	}
	
	
	public static String getDims(CNumber[][] arr) {
		return arr.length + "x" + arr[0].length;
	}
	
	
	public static CNumber[][] zeros(int rows, int cols) {
		CNumber[][] result = new CNumber[rows][cols];
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				result[i][j] = new CNumber(0);
			}
		}
		
		return result;
	}
	
	
	public static CNumber[] abs(CNumber[] arr) {
		for(int i = 0; i<arr.length; i++) {
			arr[i] = CNumber.abs(arr[i]);
		}
		
		return arr;
	}
	
	
	public static CNumber[] flatten(CNumber[][] arr) {
		CNumber[] flat = new CNumber[arr.length*arr[0].length];
		
		int count = 0;
		
		for(int i = 0; i < arr.length; i++) {
			for(int j = 0; j < arr[0].length; j++) {
				flat[count] = arr[i][j];
				count++;
			}
		}
		
		return flat;
	}
	
	
	public static CNumber[][] getSection(CNumber[][] data, int rowStart, int rowEnd, int colStart, int colEnd) {
		CNumber[][] result = new CNumber[rowEnd-rowStart][colEnd-colStart];
		
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[0].length; j++) {
				result[i][j] = data[i+rowStart][j+colStart];
			}
		}
		
		return result;
	}
	
	
	public static boolean contains(int[] arr, int value) {
		boolean result = false;
		
		for(int i : arr) {
			if(i == value) {
				result = true;
				break;
			}
		}
		
		return result;
	}
	
	
	/**
	 * Extends a 1D array to 2D along a specified axis.
	 * 
	 * @param arr - array to extend
	 * @param axis - axis along which to extend. 
	 * <br><br>
	 * If axis is 0 then a new
	 * dimension will be added along the rows and the entires of arr will
	 * become the column entries. This results in a <code>1xarr.length</code> 2D array.
	 * <br><br>
	 * If axis is 1, then a new dimension will 
	 * be added along the columns and the antries of arr will become the row
	 * entries. This results in a <code>arr.lengthx1</code> 2D array.
	 * 
	 * @return Array extended to 2D
	 */
	public static CNumber[][] extend2D(CNumber[] arr, int axis) {
		CNumber[][] result;
		
		if(axis == 0) {
			result = new CNumber[1][arr.length];
			
			result[0] = arr;
		} else if(axis == 1) {
			result = new CNumber[arr.length][1];
			
			for(int i = 0; i<arr.length; i++) {
				result[i][0] = arr[i];
			}
		} else {
			throw new IllegalArgumentException("Axis must be 0 or 1 but got " + axis);
		} 
		
		return result;
	}
	
	
	/**
	 * Extends a 1D array to 2D along a specified axis.
	 * 
	 * @param arr - array to extend
	 * @param axis - axis along which to extend. 
	 * <br><br>
	 * If axis is 0 then a new
	 * dimension will be added along the rows and the entires of arr will
	 * become the column entries. This results in a <code>1xarr.length</code> 2D array.
	 * <br><br>
	 * If axis is 1, then a new dimension will 
	 * be added along the columns and the antries of arr will become the row
	 * entries. This results in a <code>arr.lengthx1</code> 2D array.
	 * 
	 * @return Array extended to 2D
	 */
	public static CNumber[][] extend2D(double[] arr, int axis) {
		CNumber[][] result;
		
		if(axis == 0) {
			result = new CNumber[1][arr.length];
			
			for(int i = 0; i<arr.length; i++) {
				result[0][i] = new CNumber(arr[i]);
			}
		} else if(axis == 1) {
			result = new CNumber[arr.length][1];
			
			for(int i = 0; i<arr.length; i++) {
				result[i][0] = new CNumber(arr[i]);
			}
		} else {
			throw new IllegalArgumentException("Axis must be 0 or 1 but got " + axis);
		} 
		
		return result;
	}
	
	
	/**
	 * Extends a 1D array to 2D along a specified axis.
	 * 
	 * @param arr - array to extend
	 * @param axis - axis along which to extend. 
	 * <br><br>
	 * If axis is 0 then a new
	 * dimension will be added along the rows and the entires of arr will
	 * become the column entries. This results in a <code>1xarr.length</code> 2D array.
	 * <br><br>
	 * If axis is 1, then a new dimension will 
	 * be added along the columns and the antries of arr will become the row
	 * entries. This results in a <code>arr.lengthx1</code> 2D array.
	 * 
	 * @return Array extended to 2D
	 */
	public static CNumber[][] extend2D(int[] arr, int axis) {
		CNumber[][] result;
		
		if(axis == 0) {
			result = new CNumber[1][arr.length];
			
			for(int i = 0; i<arr.length; i++) {
				result[0][i] = new CNumber(arr[i]);
			}
		} else if(axis == 1) {
			result = new CNumber[arr.length][1];
			
			for(int i = 0; i<arr.length; i++) {
				result[i][0] = new CNumber(arr[i]);
			}
		} else {
			throw new IllegalArgumentException("Axis must be 0 or 1 but got " + axis);
		} 
		
		return result;
	}
	
	
	/**
	 * Computes norm of an array as if it was a vector.
	 * 
	 * @return Norm of array
	 */
	public static CNumber norm(CNumber[] arr) {
		double result = 0;
		
		for(int i = 0; i < arr.length; i++) {
			result += CNumber.multiply(arr[i], arr[i]).mag();
		}
		
		return new CNumber(result);
	}
	
	
	// Groups instances of the same number together in array
	public static CNumber[] group(CNumber[] values) {
		CNumber temp;
		double tol = 12; // Tolerance of what is considered "equal"
		
		for(int i=0; i<values.length-1; i++) {
			for(int j=i+1; j<values.length; j++) {
				if(CNumber.round(values[i], 12).equals(CNumber.round(values[j], 12))) {
					temp = values[i+1];
					values[i+1] = values[j];
					values[j] = temp;
				}
			}
		}
		
		return values;
	}
	
	
	// FOR DEVELOPMENT TESTING ONLY //
	public static void main(String[] args) {
		CNumber[][] e = 
			{{new CNumber("1"), 		new CNumber(3), 						new CNumber("14i"), 		new CNumber("5i")},
			 {new CNumber(3),			new CNumber(CNumber.IMAGINARY_UNIT), 	new CNumber("i"), 		new CNumber(32.1)},
			 {new CNumber("3+4i"), 		new CNumber("2"), 						new CNumber("3"), 		new CNumber(3, 4)},
			 {new CNumber("1 + -3i"), 	new CNumber("5i"), 						new CNumber("1 + -3i"), 	new CNumber(CNumber.PI)}};
		
		
		
		printArr(group(flatten(e)));
				
	}
}
