package linalg;

import linalg.complex_number.CNumber;

interface MatrixComparisons {
	
	/**
	 * Checks if two matrices are element-wise equivalent.
	 * If the two matrices are different shapes, this method will return false.
	 * 
	 * @param B - Matrix to compare equality with
	 * @return True if both matrices are element-wise equivalent. Otherwise, false.
	 */
	public default boolean equals(Matrix B) {
		Matrix A = (Matrix) this;
		boolean equal = true;
		
		if(A.m != B.m || A.n != B.n) { // Then we know they are not the same shape so they can't be equal
			equal = false;
			return equal;
		}
		
		for(int i = 0; i < A.m; i++) {
			for(int j=0; j < A.n; j++) {
				if(!A.entries[i][j].equals(B.entries[i][j])) { // Then we have found 
					equal = false;
					break;
				}
			}
		}
		
		return equal;
	}
	
	
	/**
	 * Compares the dimensions of two matrices. If <code>code</code> is passed zero then this method will
	 * check if both matrices have the same number of rows and columns. If <code>code</code> is passed one
	 * then this method will check if both matrices have the same number of rows only. If <code>code</code> 
	 * is passed two then this method will check if both matrices have the same number of columns only.
	 * 
	 * To check rows against columns use <code>matMultCheck(Matrix A, Matrix B)</code>
	 * 
	 * @param B - Another matrix
	 * @param code - Choice for dimension(s) along which to check equivalence 
	 * @return - true if the dimensions specified by <code>code</code> are equvalent. 
	 * Otherwise. returns false
	 */
	public default boolean sameShape(Matrix B, int code) {
		Matrix A = (Matrix) this;
		boolean result = true;
		
		if(code == 0) { // Then check for equivalent dimensions 
			if(!A.shape.equals(B.shape)) {
				result = false;
			}
		}
		else if(code == 1) { // Then check for equvalent number of rows
			if(A.numRows() != B.numRows()) {
				result = false;
			}
		}
		else if(code == 2) { // Then check for equivalent number of columns
			if(A.numCols() != B.numCols()) {
				result = false;
			}
		}
		else { // Invalid code
			throw new IllegalArgumentException("Invalid code. Expecting 0, 1, or 2");
		}
		
		return result;
	}
	
	
	/**
	 * Compares the dimensions, rows and columns, of two matrices.
	 * 
	 * @param B - Another matrix
	 * @return - True if the number of rows and number of column
	 * of the two matrices are both equal. Otherwise, returns false.
	 */
	public default boolean sameShape(Matrix B) {
		return this.sameShape(B, 0);
	}
	
	
	/**
	 * Checks if the matrix multiplication A*B can be computed. In other words, checks
	 * if A has the same number of rows as B has columns.
	 * 
	 * @param A - First Matrix
	 * @param B - Second Matrix
	 * @return True if A has the same number of rows as B has columns. Otherwise, false.
	 */
	static boolean matMultCheck(Matrix A, Matrix B) {
		return A.n == B.m;
	}


	/**
	 * Checks if this matrix has only zero entries. If so, then the matrix is the so called zero matrix
	 * and is the additive identity for matrices of the same size.
	 *
	 * @return Returns true if matrix has all zero elements. Otherwise, returns false.
	 */
	public default boolean isZero() {
		Matrix A = (Matrix) this;
		boolean result = true;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(!A.entries[i][j].equals(CNumber.ZERO)) {
					// If at any piont an element is not 0, return false.
					return false;
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * Checks if a matrix is an identity matrix.
	 *
	 * @return Returns true if the matrix in question is an identity matrix. Otherwise, returns false.
	 */
	public default boolean isI() {
		Matrix A = (Matrix) this;
		boolean result = true;
		
		if(!A.isSquare()) {
			return false;
		}
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(i == j) {
					if(!A.entries[i][j].equals(CNumber.ONE)) {
						// If at any point a diagonal element is not 1, return false.
						return false;
					}
				} else {
					if(!A.entries[i][j].equals(CNumber.ZERO)) {
						// If at any piont a non-diagonal element is not 0, return false.
						return false;
					}
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * Checks if B is the inverse of this matrix.
	 * <br><br>
	 * See {@link linalg.Matrix#inv() matrix inverse}
	 * 
	 * @return True if B is the inverse of this matrix. Otherwise, returns false.
	 */
	public default boolean isInv(Matrix B) {
		return B.equals(((Matrix) this).inv());
	}
}
