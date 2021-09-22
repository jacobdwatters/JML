package linalg;

class MatrixChecks {

	static final int SAME_DIM = 0;
	static final int SAME_COL_AND_ROW = 1;
	
	/**
	 * Compares the dimensions of two matrices. If they differ, an illegal argument exception is thrown. <br>
	 * - To check if both matrices have the same dimensions <code>type</code> should be 0 or MatrixChecks.SAME_DIM. <br>
	 * - To check if the first matrix has the same number of columns as the second matrix has rows, <code>type</code> should be 1 or MatrixChecks.SAME_COL_AND_ROW.   
	 * 
	 * @throws IllegalArgumentException If dimensions do not match rule.
	 * @param A - First matrix to check.
	 * @param B - Second matrix to check.
	 */
	static void dimensionCheck(Matrix A, Matrix B, int type) {
		if(type == SAME_DIM) {
			if(!A.sameShape(B)) { // Checks for equivalent dimensions
				throw new IllegalArgumentException(
						"Shape mismatch. Expecting matrices to have equivalent "
						+ "shapes but recieved " + A.shape() + " and " + B.shape() + "."
						);
			}
		}
		else if(type == SAME_COL_AND_ROW) {
			if(A.entries.length != B.entries.length) {
				throw new IllegalArgumentException("Number of columns in first matrix must match \n"
						+ "number of rows in seccond matrix but got " + A.shape() + " and " + B.shape() + ".");
			}
		}
		else {
			throw new IllegalArgumentException("Invalid type. Must be in [0, 1]");
		}	
	}
}
