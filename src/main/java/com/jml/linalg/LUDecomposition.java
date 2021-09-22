package com.jml.linalg;

import com.jml.linalg.complex_number.CNumber;

/**
 * A class which provides functionality for LU factorizations and variants.
 * 
 * @author Jacob Watters
 */
class LUDecomposition {
	final static int LU = 0;
	final static int LUP = 1;
	final static int LUPQ = 2;
	final static int LDU = 3;
	
	protected static Matrix[] exicute(Matrix A, int type) {
		Matrix[] result = null;
		
		switch(type) {
			case LU:
				result = LU(A);
				break;
			case LUP:
				result = LUPartialPivot(A);
				break;
			case LUPQ:
				result = LUFullPivot(A);
				break;
			case LDU:
				result = LDUDecomposition(A);
				break;
			default:
				throw new IllegalArgumentException("Unknown LU Decomposition type: " + type);
		}
		
		return result;
	}
	
	
	/**
	 * Factors a square or rectangular matrix A into a lower triangular matrix, L, and an upper tirangular matrix, U, such that A = L*U.
	 * Note: this method does not use partial pivoting.
	 * 
	 * @param A - Matrix to factor into LU
	 * @return Matrices L and U from LU factorization. This will be a Matrix array of length 2.
	 */
	protected static Matrix[] LU(Matrix A) {	
		Matrix LU = A.copy();
		Matrix[] result = new Matrix[2];
		
		// Using Gaussian elimination
		for(int j=0; j<A.m; j++) {
			for(int i=j+1; i<A.m; i++) {
				CNumber m = CNumber.divide(LU.entries[i][j], LU.entries[j][j]);
				
				for(int k=j; k<A.m; k++) {
					LU.entries[i][k] = CNumber.subtract(LU.entries[i][k], 
							CNumber.multiply(m, LU.entries[j][k]));
					LU.entries[i][j] = m;
				}
			}
		}
		
		Matrix I = Matrix.I(A.m);
		
		result[0] = LU.tril(-1).add(I);
		result[1] = LU.triu();
		
		return result;
	}
	
	
	/**
	 * TODO: Javadoc
	 * 
	 * @param A
	 * @return
	 */
	protected static Matrix[] LUPartialPivot(Matrix A) {
		Matrix LU = A.copy(),
				P = Matrix.I(A.m); // P is the row "permutation" matrix.
		Matrix[] result = new Matrix[3];
		CNumber[] col;
		int maxIndex = 0;
		
		// Using Gaussian elimination
		for(int j=0; j<A.m; j++) {
			col = LU.getCol(j); // Extract column j
			maxIndex = maxColIndex(col, j); // Find index of max value (in absolute value) in column so that the index >= j
			
			// Make the appropriate swaps in LU and P (This is the partial pivoting step)
			LU = LU.swapRows(j, maxIndex);
			P = P.swapRows(j, maxIndex);
			
			for(int i=j+1; i<A.m; i++) {
				CNumber m = CNumber.divide(LU.entries[i][j], LU.entries[j][j]);
				
				for(int k=j; k<A.m; k++) {
					LU.entries[i][k] = CNumber.subtract(LU.entries[i][k], 
							CNumber.multiply(m, LU.entries[j][k]));
					LU.entries[i][j] = m;
				}
			}
		}
		
		Matrix I = Matrix.I(A.m);
		
		result[0] = LU.tril(-1).add(I);
		result[1] = LU.triu();
		result[2] = P;
		
		return result;
	}
	
	
	protected static Matrix[] LUFullPivot(Matrix A) {
		Matrix LU = A.copy(),
				P = Matrix.I(A.m), // P is the row "permutation" matrix.
				Q = Matrix.I(A.m); // Q is the col "permutation" matrix.
		Matrix[] result = new Matrix[4];
		int[] maxIndex = {-1, -1};
		
		// Using Gaussion elimination
		for(int j=0; j<A.m; j++) {
			maxIndex = maxIndex(LU, j);
			
			// Make the appropriate swaps in LU and P (This is the partial pivoting step)
			LU = LU.swapRows(j, maxIndex[0]);
			LU = LU.swapCols(j, maxIndex[1]);
			
			P = P.swapRows(j, maxIndex[0]);
			Q = Q.swapCols(j, maxIndex[1]);
			
			for(int i=j+1; i<A.m; i++) {
				CNumber m = CNumber.divide(LU.entries[i][j], LU.entries[j][j]);
				
				for(int k=j; k<A.m; k++) {
					LU.entries[i][k] = CNumber.subtract(LU.entries[i][k], 
							CNumber.multiply(m, LU.entries[j][k]));
					LU.entries[i][j] = m;
				}
			}
		}
		
		Matrix I = Matrix.I(A.m);
		
		result[0] = LU.tril(-1).add(I);
		result[1] = LU.triu();
		result[2] = P;
		result[3] = Q;
		
		return result;
	}
	
	
	public static Matrix[] LDUDecomposition(Matrix A) {
		Matrix L, D, U;
		Matrix[] LU = LU(A), 
				LDU = new Matrix[3];
		L = LU[0];
		U = LU[1];
		
		D = U.diag();
		
		for(int i=0; i<A.m; i++) {
			for(int j=i; j<A.n; j++) {
				U.entries[i][j] = CNumber.divide(U.entries[i][j], D.entries[i][i]);
			}
		}
		
		LDU[0] = L;
		LDU[1] = D;
		LDU[2] = U;
		
		return LDU;
	}
	
	
	// A helper method for LUPartialPivot(Matrix) that finds the index of the maximum entry in a column
	static int maxColIndex(CNumber[] col, int colIndex) {
		CNumber currentMax = CNumber.MIN_VALUE;
		int index = -1;
		
		for(int i=colIndex; i<col.length; i++) {
			if(col[i].compareTo(currentMax) > 0) { // Then we have found a new current max
				currentMax = col[i];
				index = i;
			}
		}
		
		return index;
	}
	
	// A helper method for LUFullPivot(Matrix) that finds the index of the maximum entry in a submatrix
	static int[] maxIndex(Matrix matrix, int startIndex) {
		CNumber currentMax = CNumber.MIN_VALUE;
		int[] index = {-1, -1};
		
		for(int i=startIndex; i<matrix.m; i++) {
			for(int j=startIndex; j<matrix.n; j++) {
				if(matrix.get(i, j).compareTo(currentMax) > 0) {
					currentMax = matrix.get(i, j);
					index[0] = i; 
					index[1] = j;
				}
			}
		}
		
		return index;
	}
}
