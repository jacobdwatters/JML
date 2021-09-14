package linalg;

import linalg.complex_number.CNumber;

/**
 * This interface provides several methods useful for matrix operations.
 * 
 * @author Jacob Watters
 */
interface MatrixOperations {
	
	/**
	 * Performs matrix addition on two matrices of the same dimensions.
	 * 
	 * @param B - matrix to add to the instance matrix
	 * @return result of matrix addition
	 */
	 default Matrix add(Matrix B) {
		Matrix A = (Matrix) this;
		Matrix C = new Matrix(A.m, A.n);
		MatrixChecks.dimensionCheck(A, B, MatrixChecks.SAME_DIM);
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				C.entries[i][j].re = A.entries[i][j].re + B.entries[i][j].re;
				C.entries[i][j].im = A.entries[i][j].im + B.entries[i][j].im;
			}
		}
		
		return C;
	}
	
	
	/**
	 * Performs matrix subtraction on two matrices of the same dimensions.
	 * 
	 * @param B - matrix to subtract to the instance matrix
	 * @return result of matrix subtraction
	 */
	 default Matrix sub(Matrix B) {
		Matrix A = (Matrix) this;
		Matrix C = new Matrix(A.m, A.n);
		MatrixChecks.dimensionCheck(A, B, MatrixChecks.SAME_DIM);
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				C.entries[i][j].re = A.entries[i][j].re - B.entries[i][j].re;
				C.entries[i][j].im = A.entries[i][j].im - B.entries[i][j].im;
			}
		}
		
		return C;
	}
	
	
	/**
	 * Performs matrix multiplication on two matrices. The instance matrix
	 * must have the same number of columns as the rows of B.
	 * 
	 * If the instance matrix is a kxm matrix and B is a 
	 * m x n matrix then the result will be a k x n matrix.
	 * 
	 * @param B - matrix to multiply to the instance matrix
	 * @return result of matrix multiplication
	 */
	 default Matrix mult(Matrix B) {
		Matrix A = (Matrix) this;
		
		if(!MatrixComparisons.matMultCheck(A, B)) {
			throw new IllegalArgumentException("Number of columns in first matrix must match \n"
					+ "number of rows in seccond matrix but got " + A.shape() + " and " + B.shape() + ".");
		}
		
		Matrix product = new Matrix(A.m, B.n);
		
		for(int i = 0; i < product.m; i++) {
			for(int k = 0; k < A.n; k++) {
				for(int j = 0; j < product.n; j++) {
					product.entries[i][j].re += (A.entries[i][k].re * B.entries[k][j].re - A.entries[i][k].im * B.entries[k][j].im);
					product.entries[i][j].im += (A.entries[i][k].re * B.entries[k][j].im + A.entries[i][k].im * B.entries[k][j].re);
				}
			}
		}
		
		return product;
	}
	
	
	/**
	 * Performs element-wise multiplication of two matrices.
	 * 
	 * @throws IllegalArgumentException If matrices do not have the same dimension.
	 * @param B - matrix to multiply element-wise to this matrix.
	 * @return result of element-wise matrix multiplication.
	 */
	 default Matrix elemMult(Matrix B) {
		Matrix A = (Matrix) this;
		Matrix C = new Matrix(A.m, A.n);
		MatrixChecks.dimensionCheck(A, B, MatrixChecks.SAME_DIM);
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				C.entries[i][j].re = A.entries[i][j].re * B.entries[i][j].re - A.entries[i][j].im * B.entries[i][j].im;
				C.entries[i][j].im = A.entries[i][j].re * B.entries[i][j].im + A.entries[i][j].im * B.entries[i][j].re;
			}
		}
		
		return C;
	}
	
	
	
	/**
	 * Performs scalar multiplication of a matrix.
	 * 
	 * @param factor - value to multiply this matrix by.
	 * @return The scalar multiplication of the matrix and the factor.
	 */
	 default Matrix scalMult(double factor) {
		return this.scalMult(new CNumber(factor));
	}
	
	
	/**
	 * Performs scalar multiplication of a matrix.
	 * 
	 * @param factor - value to multiply matrix by.
	 * @return The scalar multiplication of the matrix and the factor.
	 */
	 default Matrix scalMult(CNumber factor) {
		Matrix A = (Matrix) this;
		Matrix result = new Matrix(A.m, A.n);
		
		for(int i = 0; i < A.m; i++) {
			for(int j = 0; j <  A.n; j++) {
				result.entries[i][j] = CNumber.multiply(A.entries[i][j], factor);
			}
		}
		
		return result;
	}
	
	
	/**
	 * Performs element-wise division on two matrices of the same dimensions.
	 * 
	 * @param B - matrix to divide element-wise the instance matrix with.
	 * @return result of element-wise matrix multiplication.
	 */
	 default Matrix elemDiv(Matrix B) {
		Matrix A = (Matrix) this;
		Matrix C = new Matrix(A.m, A.n);
		MatrixChecks.dimensionCheck(A, B, MatrixChecks.SAME_DIM);
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				C.entries[i][j].re = 	(A.entries[i][j].re * B.entries[i][j].re + A.entries[i][j].im * B.entries[i][j].im) / 
										(B.entries[i][j].re * B.entries[i][j].re + B.entries[i][j].im * B.entries[i][j].im);
				
				C.entries[i][j].im = 	(A.entries[i][j].im * B.entries[i][j].re - A.entries[i][j].re * B.entries[i][j].im) / 
										(B.entries[i][j].re * B.entries[i][j].re + B.entries[i][j].im * B.entries[i][j].im);
			}
		}
		
		return C;
	}
	
	
	/**
	 * Performs scalar division of this matrix.
	 * 
	 * @param divisor - value to divide matrix by.
	 * @return The scalar division of the matrix and the divisor.
	 */
	 default Matrix scalDiv(double divisor) {
		double factor = 1/divisor;
		return this.scalMult(new CNumber(factor));
	}
	
	
	/**
	 * Performs scalar division of this matrix.
	 * 
	 * @param divisor - value to divide matrix by.
	 * @return The scalar division of the matrix and the divisor.
	 */
	 default Matrix scalDiv(CNumber divisor) {
		CNumber factor = CNumber.divide(CNumber.ONE, divisor);
		return this.scalMult(factor);
	}
	
	
	/**
	 * Computes the Frobenius inner product of two matrices A and B, {@code <A, B>}<sub>F</sub>.
	 * 
	 * @param B - Second matrix for the Frobenius inner product.
 	 * @return the Frobenius inner product.
	 */
	 default CNumber fip(Matrix B) {
		return this.mult(B).tr();
	}
	
	
	/**
	 * Computes the matrix direct sum. That is, a block diagonal matrix containing all matrices from a set of matrices.
	 * 
	 * @param matrixList - List of matrices from which to compute the matrix direct sum.
	 * @return The result of direct summing the matrices in matrixList to this matrix.
	 */
	 default Matrix directSum(Matrix... matrixList) { // TODO: Because this is not static, include "this" in the direct sum.
		Matrix A = (Matrix) this;
		int new_m = A.m, new_n = A.n,
			current_m = 0, current_n = 0;
		
		for(int i=0; i<matrixList.length; i++) {			
			new_m += matrixList[i].m;
			new_n += matrixList[i].n;
		}
		
		
		Matrix directSum = new Matrix(new_m, new_n);
		
		for(int i=-1; i<matrixList.length; i++) {
			if(i==-1) {
				directSum.setSlice(current_m, current_n, A);
				current_m += A.m;
				current_n += A.n;
			} else {
				directSum.setSlice(current_m, current_n, matrixList[i]);
				current_m += matrixList[i].m;
				current_n += matrixList[i].n;
			}
		}
		
		return directSum;
	}
	
	
	/**
	 * Computes element wise square root of the matrix. All square roots are the positive root or, in
	 * the case of complex entries, the root with positive real part.
	 * 
	 * @return The element-wise square root of this matrix.
	 */
	 default Matrix sqrt() {
		Matrix A = ((Matrix) this).copy();
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				A.entries[i][j] = CNumber.sqrt(A.entries[i][j]);
			}
		}
		
		return A;
	}
	
	
	/**
	 * Computes absolute value, element-wise, of a matrix.
	 * If any of the matrix cells are complex, this will result
	 * in the magnitude of that value.
	 * 
	 * @return - element-wise absolute value of matrix.
	 */
	 default Matrix abs() {
		Matrix A = (Matrix) this;
		Matrix abs = new Matrix(A.m, A.n);
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				abs.entries[i][j] = CNumber.abs(A.entries[i][j]);
			}
		}
		
		return abs;
	}
	
	
	/**
	 * Transposes Matrix. Same as <code> Matrix.T() </code>
	 * 
	 * @return transpose of matrix
	 */
	 default Matrix transpose() {
		return this.T();
	}
	
	
	
	/**
	 * Transposes Matrix. Same as <code> Matrix.transpose() </code>
	 * 
	 * @return transpose of matrix
	 */
	 default Matrix T() {
		Matrix A = (Matrix) this;
		Matrix At = new Matrix(A.n, A.m);
		
	    final int BLOCK_SIZE = 16;
	    
	    // Compute transpose using blocked algorithm
	    for(int i = 0; i < A.m; i += BLOCK_SIZE) {
	        for (int j = 0; j < A.n; j += BLOCK_SIZE) {
	            // transpose the block beginning at [i,j]
	            for (int k = i; k < i + BLOCK_SIZE && k < A.m; ++k) {
	                for (int l = j; l < j + BLOCK_SIZE && l < A.n; ++l) {
	                    At.entries[l][k] = A.entries[k][l];
	                }
	            }
	        }
	    }
	    
	    return At;
	}
	
	
	/**
	 * Conjugates a matrix element-wise.
	 * 
	 * @return Conjugate of matrix
	 */
	 default Matrix conjugate() {
		Matrix A = (Matrix) this;
		Matrix Ac = new Matrix(A.m, A.n);
		
		for(int i = 0; i < Ac.m; i++) {
			for(int j = 0; j < Ac.n; j++) {
				Ac.entries[i][j] = CNumber.conjugate(A.entries[i][j]);
			}
		}
		
		return Ac;
	}
	
	
	/**
	 * Computes the conjugate transpose of this matrix. 
	 * <br><br>
	 * This method is the same as {@link #hermAdjoint() hermAdjoint()} and {@link #H() H()}. 
	 * 
	 * @return The conjugate transpose of this matrix.
	 */
	 default Matrix conjT() {
		return this.conjugate().T();
	}
	
	
	/**
	 * Computes the Hermation adjoint of a matrix. This is the transpose of the conjugate matrix.
	 * <br><br>
	 * This method is the same as {@link #conjT() conjT()}  and {@link #H() H()}. 
	 * 
	 * @return adjoint of matrix.
	 */
	 default Matrix hermAdjoint() {
		return this.conjugate().T();
	}
	
	
	/**
	 * Computes the Hermation adjoint of a matrix. This is the transpose of the conjugate matrix.
	 * <br><br>
	 * This method is the same as {@link #conjT() conjT()} and {@link #hermAdjoint() hermAdjpint()}.
	 * 
	 * @return adjoint of matrix.
	 */
	 default Matrix H() {
		return this.conjugate().T();
	}
	
	
	/**
	 * Computes determinant of matrix using reccurisive definition. This method exists
	 * for computing the determinant of non-real matrices.
	 * 
	 * @param A - Matrix to compute determinant of
	 * @return determinant of matrix A
	 */
	static CNumber detRec(Matrix A) {
		CNumber value = CNumber.ZERO;
		
		if(A.m == 1 && A.n == 1) { // Then we simply have a 1x1 matrix.
			return A.entries[0][0];
		}
		else if(A.m == 2 && A.n == 2) { // Then we have a 2x2 matrix, which is the base case
			CNumber a = A.entries[0][0];
			CNumber b = A.entries[0][1];
			CNumber c = A.entries[1][0];
			CNumber d = A.entries[1][1];
			
			value = CNumber.subtract(CNumber.multiply(a, d), CNumber.multiply(b, c));
		} 
		else {
			for(int j = 0; j < A.n; j++) {
				Matrix newA = A.removeRow(0).removeCol(j);
				
				CNumber num = new CNumber(Math.pow(-1, j));
				num = CNumber.multiply(num, A.entries[0][j]);
				
				value = CNumber.add(value, CNumber.multiply(detRec(newA), num));
			}
		}
		
		return value;
	}
	
	
	/**
	 * Computes determinant of real matrix using QR decomposition.
	 * 
	 * @param A - Matrix to compute determinant of.
	 * @return determinant of matrix A
	 */
	static CNumber detQR(Matrix A) {
		CNumber detQ, detR = CNumber.ONE;
		Matrix[] QRh = Decompose.QR(A, true);
		
		detQ = new CNumber(Math.pow(-1, QRh[2].entries[0][0].getReal()));
		
		for(int i = 0; i<QRh[1].n; i++) {
			detR = CNumber.multiply(detR, QRh[1].entries[i][i]);
		}
		
		return CNumber.multiply(detR, detQ);
	}
	
	
	/**
	 * Computes determinant of matrix. If the matrix has
	 * any complex entries, this may be a complex value.
	 * Note: Currently this method only works for real matirces.
	 * 
	 * @return determinant of matrix.
	 */
	 default CNumber det() {
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got shape " + A.shape);
		}
		
		if(A.m == 2 && A.n == 2) { // Then we have a 2x2 matrix, which we can directly compute
			CNumber a = A.entries[0][0];
			CNumber b = A.entries[0][1];
			CNumber c = A.entries[1][0];
			CNumber d = A.entries[1][1];
			
			return CNumber.subtract(CNumber.multiply(a, d), CNumber.multiply(b, c));
		}
		
		return detQR(A);
	}
	
	
	/**
	 * Stacks matrices along rows. Both matrices must have the same number of columns
	 * Also see {@link #stack(Matrix, int) stack(Matrix B, int axis)}
	 * 
	 * @param B
	 * @return
	 */
	 default Matrix stack(Matrix B) {
		return this.stack(B, 0);
	}
	
	
	/**
	 * Stacks matrices along specified axis. Axis 0 will stack matrices along the rows. Axis 1 will
	 * stack matrices along columns.
	 * 
	 * Note: To stack matrices along axis 0 they must have the same number of columns.
	 * To stack matrices along axis 1 they must have the same number of rows.
	 * 
	 * @param B - Matrix to stack
	 * @param axis - Axis along which to stack matrices.
	 * @return Returns A and B stacked along specified axis.
	 */
	 default Matrix stack(Matrix B, int axis) {
		Matrix A = (Matrix) this;
		Matrix result = null;
		
		if(axis == 0) { // Stack on rows
			if(A.entries[0].length != B.entries[0].length) { // Ensure matricies have same number or columns
				throw new IllegalArgumentException("Must have same number of columns but "
						+ "got " + A.entries[0].length + " and " + B.entries[0].length);
			}
			
			result = new Matrix(A.entries.length+B.entries.length, A.entries[0].length);
			
			for(int i = 0; i < result.entries.length; i++) {
				for(int j = 0; j < result.entries[0].length; j++) {
					if(i < A.entries.length) { // Then copy from A.entries
						result.entries[i][j] = A.entries[i][j];
					} else { // the copy from B.entries
						result.entries[i][j] = B.entries[i-A.entries.length][j];
					}
				}
			}
		} 
		else if(axis == 1) { // Stack on cols
			if(A.entries.length != B.entries.length) { // Ensure matricies have same number or rows
				throw new IllegalArgumentException("Must have same number of rows but "
						+ "got " + A.entries.length + " and " + B.entries.length);
			}
			
			result = new Matrix(A.entries.length, A.entries[0].length + B.entries[0].length);
			
			for(int i = 0; i < result.entries.length; i++) {
				for(int j = 0; j < result.entries[0].length; j++) {
					if(j < A.entries[0].length) { // Then copy from A.entries
						result.entries[i][j] = A.entries[i][j];
					} else { // the copy from B.entries
						result.entries[i][j] = B.entries[i][j-A.entries[0].length];
					}
				}
			}
		}
		else {
			throw new IllegalArgumentException("axis must be 0 or 1 but got " + axis);
		}		
		
		return result;
	}
	
	
	/**
	 * Augments two matrices. This is the same as {@link #stack(Matrix, int) stack(B, 1)}
	 * 
	 * @param B - Matrix to augment to this matrix.
	 * @return The matrix B augmented to this matrix.
	 */
	 default Matrix augment(Matrix B) {
		return this.stack(B, 1);
	}
	
	
	/**
	 * Computes row-echelon form of matrix. This will be an upper-triangular matrix.<br>
	 * 
	 * <pre>
	 * A matrix is in row-echelon form if:
	 *  - The first non-zero element in each row, called the leading entry (also called the pivot), is 1.
	 *  - The pivot of a nonzero row is always strictly to 
	 *    the right of the leading coefficient of the row above it.
	 *  - Rows with all zero elements, if any, are below rows having a non-zero element.
	 * </pre>
	 * 
	 * A matrix can be transformed into a row equivalent matrix in row-echelon form using row operations.
	 * This is done using Gaussian (Gauss-Jordan) elimination. <br><br>
	 * 
	 * Also see <code>{@link #rref() rref()}</code> for reduced row-echelon form.
	 * 
	 * @return Row-echelon form of matrix
	 */
	 default Matrix ref() {
		Matrix A = ((Matrix) this).copy();
		CNumber m, scale;
		
		int pivotRow = 0,
			pivotCol = 0;
		
		while(pivotRow<A.m && pivotCol<A.n) {
			if(!A.entries[pivotRow][pivotCol].equals(CNumber.ZERO)) { // Then we can make the pivot one.
				 scale = CNumber.divide(CNumber.ONE, A.entries[pivotRow][pivotCol]);
				 
				 for(int k=pivotCol; k<A.n; k++) { // scale the whole row
					 A.entries[pivotRow][k] = CNumber.multiply(A.entries[pivotRow][k], scale); 
				 }
			}
			
			for(int i=pivotRow+1; i<A.m; i++) {
				m = A.entries[i][pivotCol];

				for(int k=pivotCol; k<A.n; k++) {
					A.entries[i][k] = CNumber.subtract(A.entries[i][k], 
							CNumber.multiply(A.entries[pivotRow][k], m));
				}
				
				
				/*
				 * This insures the entries to the left of the pivot are zero. 
				 * They may be a very small (in absolute value) non-zero value
				 * resulting from errors in floating point arithmetic.
				 */
				A.entries[i][pivotCol] = CNumber.ZERO;
			}
			
			pivotRow++;
			pivotCol++;
		}
		
		return A;
	}
	
	
	/**
	 * Computes reduced row-echelon form of matrix.<br>
	 * 
	 * <pre>
	 * A matrix is in reduced row-echelon form if:<br>
	 *  - It is in row-echelon form. This is,
	 *      ~ The first non-zero element in each row, called the leading entry (also called the pivot), is 1.
	 *      ~ The pivot of a nonzero row is always strictly to 
	 *        the right of the leading coefficient of the row above it.
	 *      ~ Rows with all zero elements, if any, are below rows having a non-zero element.<br>
	 *  - The pivot in each row is the only non-zero entry in its column.
	 * </pre>
	 * 
	 * A matrix can be transformed into a row eqivalent matrix in reduced row-echelon form using row operations.
	 * This is done using Gaussian (Gauss-Jordan) elimination. <br><br>
	 * 
	 * @param partialPivoting - Falg for use of partial pivoting.
	 * <pre>
	 *  - If true then the rref will be computed using partial pivoting.
	 * 	    ~ This is equivalent to the method {@link #rref() rref()}.
	 *  - If false then the rref will be computed WITHOUT using partial pivoting.
	 * </pre>
	 * @return
	 */
	 default Matrix rref(boolean partialPivoting) {
		if(partialPivoting) return rref();
		else return rrefNoPivot();
	}
	
	
	/**
	 * Computes reduced row-echelon form of matrix. This is done using partial pivoting.<br>
	 * 
	 * <pre>
	 * A matrix is in reduced row-echelon form if:<br>
	 *  - It is in row-echelon form. This is,
	 *      ~ The first non-zero element in each row, called the leading entry (also called the pivot), is 1.
	 *      ~ The pivot of a nonzero row is always strictly to 
	 *        the right of the leading coefficient of the row above it.
	 *      ~ Rows with all zero elements, if any, are below rows having a non-zero element.<br>
	 *  - The pivot in each row is the only non-zero entry in its column.
	 * </pre>
	 * 
	 * A matrix can be transformed into a row eqivalent matrix in reduced row-echelon form using row operations.
	 * This is done using Gaussian (Gauss-Jordan) elimination. <br><br>
	 * 
	 * Also see <code>{@link #ref() ref()}</code> for row-echelon form.
	 * 
	 * @return Row-echelon form of matrix.
	 */
	 default Matrix rref() {
		Matrix A = ((Matrix) this).copy();
		CNumber mult, scale, currentMax;
		int maxIndex;
		
		int pivotRow = 0,
			pivotCol = 0;
		
		while(pivotRow<A.m && pivotCol<A.n) {
			maxIndex = pivotRow;
			currentMax = A.entries[pivotRow][pivotCol];
			
			for(int i=pivotRow; i<A.m; i++) { // Find the maximum entry in the pivot column (at or below the pivot ).
				if(A.entries[i][pivotCol].nearZero(1.0E-12)) { 
					
					/* 
					 * If a number is very close to zero, assume it is supposed to be zero.
					 * This protects against trying to manipulate a column which
					 * is already in the correct form.
					 */
					A.entries[i][pivotCol] = CNumber.ZERO;
				}
				else if(A.entries[i][pivotCol].compareTo(currentMax) > 0) {
					maxIndex = i;
					currentMax = A.entries[i][pivotCol];
				}
			}
			
			if(!A.entries[maxIndex][pivotCol].equals(CNumber.ZERO)) { // Check that the maximum absolute value is not zero.
				if(pivotRow != maxIndex) {
					A = A.swapRows(pivotRow, maxIndex); // Make the row with the largest value in the pivot column the pivot for this row.
				}
				
				scale = CNumber.divide(CNumber.ONE, A.entries[pivotRow][pivotCol]);
				 
				for(int k=pivotCol; k<A.n; k++) { // scale the whole row so that the pivot is 1
					A.entries[pivotRow][k] = CNumber.multiply(A.entries[pivotRow][k], scale); 
				}
				
				for(int i=0; i<A.m; i++) {
					mult = A.entries[i][pivotCol];
					
					if(pivotRow != i) {
						A.entries[i][pivotCol] = CNumber.ZERO;
						
						for(int k=pivotCol+1; k<A.n; k++) {
							
							CNumber value = CNumber.subtract(A.entries[i][k], 
									CNumber.multiply(A.entries[pivotRow][k], mult));
							A.entries[i][k] = value;
						}
						
					}
				}
				
				pivotRow++;
				pivotCol++;
			}
			else { // Then we do not have a pivot for this column (i.e. the column is all zeros).
				pivotCol++;
			}
		}
		
		return A;
	}
	
	
	/**
	 * Computes reduced row-echelon form of matrix. 
	 * This is done WITHOUT using partial pivoting.<br>
	 * 
	 * <pre>
	 * A matrix is in reduced row-echelon form if:<br>
	 *  - It is in row-echelon form. This is,
	 *      ~ The first non-zero element in each row, called the leading entry (also called the pivot), is 1.
	 *      ~ The pivot of a nonzero row is always strictly to 
	 *        the right of the leading coefficient of the row above it.
	 *      ~ Rows with all zero elements, if any, are below rows having a non-zero element.<br>
	 *  - The pivot in each row is the only non-zero entry in its column.
	 * </pre>
	 * 
	 * A matrix can be transformed into a row eqivalent matrix in reduced row-echelon form using row operations.
	 * This is done using Gaussian (Gauss-Jordan) elimination. <br><br>
	 * 
	 * Also see <code>{@link #ref() ref()}</code> for row-echelon form.
	 * 
	 * @return Row-echelon form of matrix.
	 */
	default Matrix rrefNoPivot() {
		Matrix A = ((Matrix) this).copy();
		CNumber m, scale;
		
		int pivotRow = 0,
			pivotCol = 0;
		
		while(pivotRow<A.m && pivotCol<A.n) {
			if(!A.entries[pivotRow][pivotCol].equals(CNumber.ZERO)) { 
				scale = CNumber.divide(CNumber.ONE, A.entries[pivotRow][pivotCol]);
				 
				for(int k=pivotCol; k<A.n; k++) { // scale the whole row
					A.entries[pivotRow][k] = CNumber.multiply(A.entries[pivotRow][k], scale); 
				}	
			}
				
			for(int i=0; i<A.m; i++) {
				m = A.entries[i][pivotCol];
				
				if(pivotRow != i) {
					for(int k=pivotCol; k<A.n; k++) {
						A.entries[i][k] = CNumber.subtract(A.entries[i][k], 
								CNumber.multiply(A.entries[pivotRow][k], m));
					}
				}
			}
			
			pivotRow++;
			pivotCol++;
		}
		
		return A;
	}
	
	
	/**
	 * Computes reduced extended row-echelon form of matrix. That is, a Matrix with
	 * the same number of rows is {@link #augment(Matrix) augmented} with this matrix and
	 * then this augmented matrix is put into {@link #rref() reduced row-echelon form}.
	 * 
	 * @return Returns extended row-echelon form of this matrix.
	 */
	 default Matrix erref() {
		Matrix A = (Matrix) this;
		Matrix I = Matrix.I(A.m);
		Matrix Aug = A.augment(I);
		
		return Aug.rref();
	}

	
	/**
	 * Computes the trace of square matrix. That is, the sum 
	 * of the entries along the principle diagonal.
	 * <br><br>
	 * This method is the same as {@link #tr() tr()}.
	 * 
	 * @return trace of this matrix.
	 */
	 default CNumber trace() {
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Expecting matrix to be square but got " + A.shape);
		}
		
		CNumber result = A.entries[0][0];
		
		for(int i = 1; i < A.n; i++) {
			result = CNumber.add(result, A.entries[i][i]);
		}
		
		return result;
	} 
	
	
	/**
	 * Computes the trace of a square matrix. That is, the sum 
	 * of the entries along the principle diagonal.
	 * <br><br>
	 * This method is the same as {@link #trace() trace()}.
	 * 
	 * @return trace of this matrix.
	 */
	 default CNumber tr() {
		return this.trace();
	}
	
	
	/**
	 * The rank of a matrix A is the dimension of the vector space generated (or spanned) by its columns.
	 * This is always an integer.
	 * This corresponds to the maximal number of linearly independent columns of A. 
	 * This, in turn, is identical to the dimension of the vector space spanned by its rows
	 * 
	 * @return Returns the rank of this matrix.
	 */
	// TODO: Should be switched to rank revealing QR decomposition as it is more numerically stable.
	 default int rank() {
		Matrix A = (Matrix) this;
		Matrix rrefA = A.rref();
		int rank = A.m;
		
		for(int i=rrefA.m-1; i>=0; i--) {
			if(rrefA.getRowAsVector(i).isZero()) {
				rank--;			
			} else { 
				// Then we have no more zero rows as rrefA is in reduced row echelon form.
				break;
			}
		}
		
		return rank;
	}
	
	
	/**
	 * The rank of a matrix A is the dimension of the vector space spanned by the nullspace of this matrix.
	 * The nullify is always an integer.
	 * 
	 * @return Returns the rank of this matrix.
	 */
	 default int nullity() {
		int rank = this.rank();
		return ((Matrix) this).m - rank;
	}
	
	
	/**
	 * Computes the matrix inverse if it exists. This is done by first computing the {@link linalg.Decompose#QR(Matrix) QR decomposition}
	 * The inverse of a Matrix A is A<sup>-1</sup> satisfying AA<sup>-1</sup>=I where I is the appropriately sized Identity matrix.
	 * 
	 * @return The inverse of this matrix.
	 */
	 default Matrix inverse() {
		Matrix A = (Matrix) this;
		
		if(!A.isSingular()) {
			throw new IllegalArgumentException("Matrix is singular.");
		}

		Matrix[] QR = Decompose.QR(A); // Compute the QR decomposition of A. This will be used to compute the inverse.
		CNumber detR = CNumber.ONE;
		
		for(int i = 0; i<QR[1].n; i++) { // Computing the determinant of R
			detR = CNumber.multiply(detR, QR[1].entries[i][i]);
		}
		
		if(CNumber.round(detR, 10).re == 0) {
			/* Then we know the matrix is singular.
			 * We know this by exploiting properties of the orthogonal matrix Q and the upper triangular matrix R.
			 * The determinant of an orthogonal matrix is either -1 or 1. The determinant of an upper triangular matrix R is the product
			 * of the diagonals. Since A=QR det(A) = det(QR) = det(Q)*det(R). So if the determinant of R is zero, then the matrix A must be singular.
			 * 
			 * The isSingular() method is not used here because it computes the QR decomposition to check if the matrix is singular. However, we have already done that, 
			 * so it would be redundant to call the isSingular() method and recompute the QR decomposition again.
			 */
			throw new IllegalArgumentException("Matrix is singular, thus can not be inverted.");
		}
		
		// Here we compute and return A^-1 = (R^-1)*(Q^-1)
		return inverseTriU(QR[1]).mult(QR[0].H());
	}
	
	
	/**
	 * Computes the matrix inverse if it exists. This is done by first computing the {@link linalg.Decompose#QR(Matrix) QR decomposition}
	 * The inverse of a Matrix A is A<sup>-1</sup> satisfying AA<sup>-1</sup>=I where I is the appropriately sized Identity matrix.
	 * 
	 * @return The inverse of this matrix.
	 */
	 default Matrix inv() {
		return this.inverse();
	}
	
	
	/**
	 * Helper method to invert an upper triangular matrix. <br>
	 * We can compute the inverse of an n-by-n upper triangular matrix U by solving the n systems...<br><br>
	 * 
	 * Ux=I<sub>i</sub><br>
	 * 
	 * where 1<=i<=n, I is the n-by-n identity matrix, I<sub>i</sub> is the ith column of I.<br><br>
	 * 
	 * Each solution becomes a column in the inverse <sup>-1</sub>.
	 * Because U is upper triangular, we can solve these n systems using backsolve.
	 * 
	 * @param U - U is a square upper triangular matrix
	 * @return Returns the inverse of U, denoted U<sup>-1</sub>
	 */
	static Matrix inverseTriU(Matrix U) {
		Matrix Uinv = new Matrix(U.m, U.m);
		Matrix I = Matrix.I(U.m);
		
		if(!U.isTriU()) {
			throw new IllegalArgumentException("U must be upper triangular.");
		}
		
		for(int i=0; i<U.m; i++)  {
			// U is upper triangular, so to solve the system we simply need to use backsolve function.
			Uinv.setCol(Solvers.backSolve(U, I.getColAsVector(i)), i); 
		}
		
		return Uinv;
	}
	
	
	/**
	 * Creates a new matrix that contains the reciprocals of this matrix
	 * 
	 * @return new matrix that contains the reciprocals of this matrix
	 */
	 default Matrix recep() {
		Matrix A = new Matrix((Matrix) this).copy();
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				A.entries[i][j] = CNumber.inv(A.entries[i][j]);
			}
		}
		
		return A;
	}
	
	
	 static void main(String[] args) {

		
		int[][] b = {{1, 3, 3},
				 	 {0, 5, 6},
				 	 {0, 8, 9}};
		
		double[][] c = {{ -19.1635403109,        5,               8,               7		},
				  		{       5,         -19.1635403109,        2,               8		},
						{       7,               5,         -15.1635403109,        6		},
						{       5,               4,               4,         -13.1635403109 }};
		
		CNumber[][] bc = {{new CNumber("2+i"), new CNumber("3")},
				  		  {new CNumber("1"), new CNumber("-i")}};

		Matrix A = new Matrix(bc);
		Matrix B = new Matrix(b);
		Matrix C = new Matrix(c);
		
		Matrix.print("C:\n", B.sqrt(), "\n\n");
		Matrix.print("C:\n", B, "\n\n");

		Matrix.print("rref:\n", A.directSum(B, C), "\n\n");
	}
}

