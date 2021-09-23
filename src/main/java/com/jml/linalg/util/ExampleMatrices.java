package com.jml.linalg.util;

import com.jml.linalg.Matrix;
import com.jml.linalg.complex_number.CNumber;


/**
 * This interface just contains some different matrices for testing purposes.<br>
 *
 * <pre>
 *     Matrices:
 *     		- {@link #MATRIX_0}
 *     		- {@link #MATRIX_1}
 *     		- {@link #MATRIX_2}
 *     		- {@link #MATRIX_3}
 *     		- {@link #MATRIX_4}
 *     		- {@link #MATRIX_5}
 *     		- {@link #MATRIX_6}
 *     		- {@link #MATRIX_7}
 *     		- {@link #MATRIX_8}
 * </pre>
 *
 * @author Jacob Watters
 */
public class ExampleMatrices {
	
	final static double[][] a = 
		{{11, 		6.011, 		4.656841},
		 {6.011, 	4.656841, 	4.11907321},
		 {4.656841, 4.11907321, 3.9225424}};
	
	final static int[][] b = 
		{{1, 2, 3, 1, 2, 3, 4, 5, 6},
	 	 {4, 5, 6, 3, 4, 5, 7, 8, 2},
	 	 {1, 5, 5, 2, 6, 7, 9, 0, 1},
	 	 {3, 4, 5, 2, 6, 7, 8, 9, 2},
	 	 {1, 1, 1, 3, 4, 7, 8, 9, 1},
	 	 {3, 4, 7, 8, 3, 1, 2, 3, 4},
	 	 {3, 5, 6, 8, 1, 3, 5, 9, 1},
	 	 {0, 4, 5, 3, 2, 0, 1, 2, 3},
	 	 {1, 4, 5, 1, 7, 0, 5, 3, 1}};
	
	final static double[][] c = {{2.4}};
	
	final static String[][] d = 
		{{"5", 		"2-2.3i", 	"4.1 + -3.1i"},
		 {"3i", 	"-i", 		"4.1"},
		 {"0+0i",	"0", 		"1"}};
	
	final static CNumber[][] e = 
		{{new CNumber("16+2.1i"), 	new CNumber(CNumber.E), 				new CNumber("14i"), 		new CNumber("12.3")},
		 {new CNumber(3),			new CNumber(CNumber.IMAGINARY_UNIT), 	new CNumber("3+5i"), 	new CNumber(32.1)},
		 {new CNumber("2+14.2i"), 	new CNumber("4 - -2i"), 				new CNumber("1 - 2i"), 	new CNumber(3, 4)},
		 {new CNumber("13-4.5i"), 	new CNumber("115+44.3i"), 			new CNumber("1 + -3i"), 	new CNumber(CNumber.PI)}};
	
	final static int[][] f = {{1,  2,  3,  4,  5 },
							  {6,  7,  8,  9,  10},
							  {11, 12, 13, 14, 15}};
	
	final static int[][] g = {{1, 8, 9},
							  {0, 8, 7},
							  {0, 0, 5}};
	
	final static int[][] h = {{1,  2,  3 },
							  {4,  5,  6 },
							  {7,  8,  9 },
							  {10, 11, 12},
							  {13, 14, 15}};
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[]]
	 * </pre>
	 */
	public static final Matrix MATRIX_0 = new Matrix();
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[11,       6.011,      4.656841  ] 
	 *  [6.011,    4.656841,   4.11907321] 
	 *  [4.656841, 4.11907321, 3.9225424 ]] 
	 * </pre>
	 */
	public static final Matrix MATRIX_1 = new Matrix(a);
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[1, 2, 3, 1, 2, 3, 4, 5, 6] 
	 *  [4, 5, 6, 3, 4, 5, 7, 8, 2] 
	 *  [1, 5, 5, 2, 6, 7, 9, 0, 1] 
	 *  [3, 4, 5, 2, 6, 7, 8, 9, 2] 
	 *  [1, 1, 1, 3, 4, 7, 8, 9, 1] 
	 *  [3, 4, 7, 8, 3, 1, 2, 3, 4] 
	 *  [3, 5, 6, 8, 1, 3, 5, 9, 1] 
	 *  [0, 4, 5, 3, 2, 0, 1, 2, 3] 
	 *  [1, 4, 5, 1, 7, 0, 5, 3, 1]]
	 * </pre>
	 */
	public static final Matrix MATRIX_2 = new Matrix(b);
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[2.4]]
	 * </pre>
	 */
	public static final Matrix MATRIX_3 = new Matrix(c);
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[5,  2-2.3i, 4.1-3.1i]
	 *  [3i, -i,     4.1     ]
	 *  [0,  0,      1       ]]
	 * </pre>
	 */
	public static final Matrix MATRIX_4 = new Matrix(d);
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[16 + 2.1i,  2.718281828459045,  14i,     12.3             ] 
	 *  [3,          i,                  3 + 5i,  32.1             ] 
	 *  [2 + 14.2i,  4 + 2i,             1 - 2i,  3 + 4i           ] 
	 *  [13 - 4.5i,  115 + 44.3i,        1 - 3i,  3.141592653589793]] 
	 * </pre>
	 */
	public static final Matrix MATRIX_5 = new Matrix(e);
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[1,  2,  3,  4,  5 ]
	 *  [6,  7,  8,  9,  10]
	 *  [11, 12, 13, 14, 15]]
	 * </pre>
	 */
	public static final Matrix MATRIX_6 = new Matrix(f);
	
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[1 8 9]
	 *  [0 8 7]
	 *  [0 0 5]]
	 * </pre>
	 */
	public static final Matrix MATRIX_7 = new Matrix(g);
	
	/**
	 * <p> Matrix contents: </p>
	 * 
	 * <pre>
	 * [[1  2  3 ]
	 *  [4  5  6 ]
	 *  [7  8  9 ]
	 *  [10 11 12]
	 *  [13 14 15]]
	 *  </pre>
	 */
	public static final Matrix MATRIX_8 = new Matrix(h);
}
