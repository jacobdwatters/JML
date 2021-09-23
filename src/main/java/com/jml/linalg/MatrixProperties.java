package com.jml.linalg;

import com.jml.linalg.complex_number.CNumber;
import com.jml.linalg.util.LinAlgArrayUtils;

import java.util.ArrayList;

/**
 * This interface contains several methods involving properties of a matrix.
 * 
 * @author Jacob Watters
 */
interface MatrixProperties {
	
	/**
	 * Number of columns in this matrix
	 * 
	 * @return Number of rows in matrix.
	 */
	default int numCols() { 
		Matrix A = (Matrix) this;
		return A.n;
	}

	
	/**
	 * Number of rows in this matrix.
	 * 
	 * @return Number of columns in matrix.
	 */
	default int numRows() { 
		Matrix A = (Matrix) this;
		return A.m; 
	}
	
	
	/**
	 * Shape of this matrix i.e. number of rows and columns.
	 * 
	 * @return Returns shape of this matrix shape as String e.g. "m x n".
	 */
	 default String shape() {
		Matrix A = (Matrix) this;
		return A.shape;
	}
	
	
	/**
	 * Checks if the matrix is square. That is, does the matrix have the same number of rows and columns?
	 * 
	 * @return True if the matrix is square, false otherwise.
	 */
	 default boolean isSquare() {
		Matrix A = (Matrix) this;
		return (A.m==A.n);
	}
	
	
	
	/**
	 * Checks if the matrix is empty. That is, does the matrix have exactly zero entries?
	 * 
	 * @return True if the matrix is empty, false otherwise.
	 */
	 default boolean isEmpty() {
		Matrix A = (Matrix) this;
		return (A.m==0 && A.n==0);
	}
	
	
	/**
	 * Finds the minimum value in the matrix. If the matrix is complex, the value with the smallest magnitude will be returned.
	 * If the matrix is real, the smallest real number will be returned.<br>
	 * Also see {@link #minReal() minReal()} and {@link #minComplex() minComplex()} 
	 * 
	 * @return minimum value of this matrix
	 */
	 default CNumber min() {
		CNumber min;

		if(this.isReal()) {
			min = new CNumber(this.minReal());
		} else {
			min = this.minComplex();
		}
		
		return min;
	}
	
	
	/**
	 * Finds the minimum real value in the matrix.
	 * If the matrix contains non-real values, the imaginary component will be ignored.
	 * <br>
	 * Also see {@link #min() min()} and {@link #minComplex() minComplex()} 
	 * 
	 * @return Returns minimum real value of this matrix.
	 */
	// TODO: Complex values should be completely ignored in general.
	 default double minReal() {
		double currentMin = Double.MAX_VALUE;
		Matrix A = (Matrix) this;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(A.entries[i][j].compareToReal(currentMin) < 0) {
					currentMin = A.entries[i][j].re;
				}
			}
		}
		
		return currentMin;
	}
	
	
	/**
	 * Finds the maximum value in the matrix. If the matrix is complex, the value with the largest magnitude will be returned.
	 * If the matrix is real, the largest real number will be returned.<br>
	 * Also see {@link #maxReal() maxReal()} and {@link #maxComplex() maxComplex()} 
	 * 
	 * @return Returns maximum value of this matrix.
	 */
	 default CNumber max() {
		CNumber max;
		
		if(this.isReal()) {
			max = new CNumber(this.maxReal());
		}
		else {
			max = this.maxComplex();
		}
		
		return max;
	}
	
	
	/**
	 * Finds the maximum real value in the matrix.
	 * If the matrix contains any non-real entries, the imaginary component will be ignored. 
	 * <br>
	 * Also see {@link #max() max()} and {@link #maxComplex() maxComplex()} 
	 * 
	 * @return Returns maximum real value of this matrix.
	 */
	// TODO: Complex values should be completely ignored in general.
	 default double maxReal() {
		double currentMax = Double.MIN_VALUE;
		
		Matrix A = (Matrix) this;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(A.entries[i][j].compareToReal(currentMax) > 0) {
					currentMax = A.entries[i][j].re;
				}
			}
		}
		
		return currentMax;
	}
	
	
	/**
	 * Finds value with minimum magnitude.
	 * Also see {@link #min() min()} and {@link #minReal() minReal()} 
	 * 
	 * @return Returns value with minimum magnitude in this matrix.
	 */
	 default CNumber minComplex() {
		CNumber currentMin = CNumber.MAX_VALUE;
		Matrix A = (Matrix) this;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(A.entries[i][j].compareToReal(currentMin) < 0) {
					currentMin = A.entries[i][j];
				}
			}
		}
		
		return currentMin;
	}
	
	
	/**
	 * Finds value with maximum magnitude.
	 * Also see {@link #max() max()} and {@link #maxReal() maxReal()} 
	 * 
	 * @return Returns value with maximum magnitude in this matrix.
	 */
	 default CNumber maxComplex() {
		CNumber currentMax = CNumber.MIN_VALUE;
		Matrix A = (Matrix) this;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(A.entries[i][j].compareTo(currentMax) > 0) {
					currentMax = A.entries[i][j];
				}
			}
		}
		
		return currentMax;
	}
	
	
	/**
	 * Checks if matrix is real. That is, the matrix only has real entries.
	 * 
	 * @return True if matrix has no complex entries. Otherwise, false.
	 */
	 default boolean isReal() {
		Matrix A = (Matrix) this;
		boolean result = true;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(A.entries[i][j].isComplex()) {
					result = false;
					break;
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * Checks if matrix has at least one complex entry.
	 * 
	 * @return True if matrix has at least one non-real entry. Otherwise, false.
	 */
	 default boolean isComplex() {
		Matrix A = (Matrix) this;
		boolean result = false;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(A.entries[i][j].isComplex()) {
					result = true;
					break;
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * Checks if a given matrix is a column or row vector. A column vector will be a 
	 * a matrix of shape <code>mx1</code> and a row vector will be a matrix of shape
	 * <code>1xn</code>
	 * 
	 * @return True if matrix is a column or row vector, otherwise returns false.
	 */
	 default boolean isVector() {
		Matrix A = (Matrix) this;
		boolean result = false;
		
		if(A.m == 1 || A.n == 1) {
			result = true;
		}
		
		return result;
	}
	
	
	/**
	 * Checks if a matrix is a vector and if so, what kind of vector (i.e row or column vectors).
	 * 
	 * @return 0 if the matrix is NOT a vector, 1 if the matrix is a row vector,
	 * 2 if the matrix is a column vector, and 3 if the matrix is of shape 1x1
	 */
	 default int vectorType() {
		Matrix A = (Matrix) this;
		int result = 0;
		
		if(A.m == 1 && A.n > 1) { // then we have a row vector
			result = Vector.ROW_VECTOR;
		}
		else if(A.n == 1 && A.m > 1) { // then we have a column vector
			result = Vector.COLUMN_VECTOR;
		} 
		else if(A.m == 1 && A.n == 1) { // then we have a 1x1 matrix 
			result = 3;
		}
		
		return result;
	}
	
	
	/**
	 * Checks if matrix is self-adjoint. That is, if the matrix is equal to its own
	 * conjugate transpose.<br><br>
	 * 
	 * Same as {@link #isHermation() isHermation()}.
	 * 
	 * @return True if the matrix is self-adjoint. Otherwise, returns false.
	 */
	 default boolean isSelfAdjoint() {
		return this.isHermation();
	}
	
	
	/**
	 * Checks if matrix is hermation. That is, if the matrix is equal to its own
	 * conjugate transpose.<br><br>
	 * 
	 * Same as {@link #isSelfAdjoint() isSelfAdjoint()}.
	 * 
	 * @return True if the matrix is hermation. Otherwise, returns false.
	 */
	 default boolean isHermation() {
		Matrix A = (Matrix) this;
		return A.H().equals(A);
	}
	
	
	/**
	 * Checks if a matrix is symmetric. 
	 * <br><br>
	 * For an square matrix A, 
	 * A is symmetric if and only if <code>A[i][j] = A[j][i]</code> for all i, j. 
	 * That is, if A is equal its own transpose, then it is symmetric.<br><br>
	 * 
	 * Also see {@link #isSkewSymmetric() isSkewSymmetric()} and
	 * {@link #isSymmetric(String) isSymmetric(String skewOption)}
	 * 
	 * @return True if this matrix is symmetric, false otherwise. 
	 */
	 default boolean isSymmetric() {
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix Must be square but got size " + A.shape);
		}
		
		return A.equals(A.T());
	}
	
	
	/**
	 * Checks if a matrix is skew-symmetric. 
	 * <br><br>
	 * For an square matrix A, 
	 * A is skew-symmetric if and only if <code>A[i][j] = -A[j][i]</code> for all i, j. 
	 * That is, if A is equal to the negative of its own transpose, then it is skew-symmetric.<br><br>
	 * 
	 * Also see {@link #isSymmetric() isSymmetric()} and
	 * {@link #isSymmetric(String) isSymmetric(String skewOption)}
	 * 
	 * @return True if this matrix is symmetric, false otherwise. 
	 */
	 default boolean isSkewSymmetric() {
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix Must be square but got size " + A.shape);
		}
		
		return (A.equals(A.scalMult(-1).T()));
	}
	
	
	/**
	 * Checks if a matrix is symmetric or skew-symmetric depending on the skewOption.
	 * <br><br>
	 * - If skewOption is passed "skew", this method is identical to 
	 * {@link #isSkewSymmetric() isSkewSymmetric()}.<br>
	 * 
	 * - If skewOption is passed "no-skew", this method is identical to 
	 * {@link #isSymmetric() isSymmetric()}.
	 * 
	 * <br><br>
	 * For an square matrix A, 
	 * A is symmetric if and only if <code>A[i][j] = A[j][i]</code> for all i, j. 
	 * That is, if A is equal to A transpose.
	 * <br><br>
	 * A is skew-symmetric if and only if <code>A[i][j] = -A[j][i]</code> for all i, j. 
	 * That is, if A is equal to -A transpose.
	 * 
	 * 
	 * @param skewOption - String to indicate whether to check for symmetry or
	 * 		skew-symmetry. Specify skewOption as "skew" to check for skew-symmetry or
	 * 		"no-skew" to check for regular symmetry.
	 * 					
	 * @return True if the matrix is symmetric/skew-symmetric. Otherwise, returns false.
	 */
	 default boolean isSymmetric(String skewOption) {
		Matrix A = (Matrix) this;
		boolean result = false;
		
		if(skewOption.equalsIgnoreCase("skew")) {
			result = A.isSkewSymmetric();
			
		} else if(skewOption.equalsIgnoreCase("no-skew")) {
			result = A.isSymmetric();
		}
		
		return result;
	}
	
	
	/**
	 * Checks if this matrix is orthogonal. A square matrix <code>Q</code> is
	 * orthogonal if and only if <code>QQ^T = I</code> where I is the appropriately sized
	 * identity matrix.  
	 * <br><br>
	 * Please note, this method only checks if <code>QQ^T = I</code>. If the matrix is complex
	 * you may be looking for {@link #isUnitary() isUnitary()} which checks if a matrix is unitary or <code>QQ^* = I</code> where
	 * <code>Q^*</code> is the conjugate transpose of <code>Q</code>. The <code>isUnitary()</code> method will behave the same as 
	 * this method for real matrices.
	 * 
	 * @return True if the matrix is orthogonal. Otherwise, returns false.
	 */
	 default boolean isOrthogonal() {
		Matrix A = ((Matrix) this);
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		if(!A.isReal()) {
			throw new IllegalArgumentException("Matrix must be real. For complex or real matrices, use isUnitary()" + A.shape);
		}
		
		Matrix AAT = A.mult(A.T()).round(13);
		
		return AAT.equals(Matrix.I(A.m));
	}
	
	
	/**
	 * Checks if this matrix is unitary. A square matrix <code>Q</code> is
	 * unitary if and only if <code>QQ^* = I</code> where <code>I</code> is the appropriately sized
	 * identity matrix and <code>Q^*</code> is the conjugate transpose. 
	 * <br><br>
	 * For real matrices, this method is the same as {@link #isOrthogonal() isOrthogonal()}.
	 * 
	 * @return True if the matrix is unitary. Otherwise, returns false.
	 */
	 default boolean isUnitary() {
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		Matrix AATC = A.mult(A.conjT()).round(13);
		
		return AATC.equals(Matrix.I(A.m));
	}
	
	
	/**
	 * Checks if a matrix is Triangular.<br>
	 * A triangular matrix has all zeros above and/or
	 * below the principle diagonal.<br><br>
	 * - Diagonal: A square matrix is diagonal if every element above 
	 * and below the principle diagonal is zero.<br>
	 * - Lower Triangular: A square matrix is lower triangular if
	 * every element above the principle diagonal is zero.<br>
	 * - Upper Triangular: A square matrix is upper triangular 
	 * if every element below the principle diagonal is zero
	 * <br><br>
	 * See <br>
	 * - {@link #isTriU() isTriU()}<br>
	 * - {@link #isTriL() isTriL()}<br>
	 * - {@link #isDiagonal() isDiagonal()}
	 * 
	 * @return - 1 if matrix is not triangular. <br>
	 * 		- 0 if Matrix is diagonal.<br>
	 * 		- 1 if Matrix is lower triangular.<br>
	 * 		- 2 if Matrix is upper triangular.
	 */
	 default int isTri() {
		int result = -1;
		Matrix A = (Matrix) this;
		
		if(A.isDiagonal()) {
			result = 0;
		} else if(A.isTriL()) {
			result = 1;
		} else if(A.isTriU()) {
			result = 1;
		}
		
		return result;
	}
	
	
	/**
	 * Checks if matrix is upper triangular. A square matrix is upper triangular 
	 * if every element below the principle diagonal is zero.
	 * 
	 * @return True if this matrix is upper triangular. Otherwise, returns false.
	 */
	 default boolean isTriU() {
		boolean result = true;
		Matrix A = (Matrix) this;
		
		for(int j=0; j<A.n-1; j++) {
			for(int i=j+1; i<A.m; i++) {
				if(!A.entries[i][j].equals(CNumber.ZERO)) {
					result = false;
					return result;
				}
			}
		}

		return result;
	}
	
	
	/**
	 * Checks if matrix is lower triangular. A square matrix is lower triangular if
	 * every element above the principle diagonal is zero.
	 * 
	 * @return True if this matrix is lower triangular. Otherwise, returns false.
	 */
	 default boolean isTriL() {
		boolean result = true;
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		for(int i=0; i<A.m-1; i++) {
			for(int j=i+1; j<A.n; j++) {
				if(!A.entries[i][j].equals(CNumber.ZERO)) {
					result = false;
					return result;
				}
			}
		}

		return result;
	}
	
	
	/**
	 * Checks if this matrix is diagonal. A matrix is diagonal if every element above 
	 * and below the principle diagonal is zero.
	 * 
	 * @return True if this matrix is diagonal. Otherwise, returns false.
	 */
	 default boolean isDiagonal() {
		Matrix A = (Matrix) this;
		
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		return (A.isTriL() && A.isTriU());
	}
	
	
	/**
	 * Checks to see if a matrices rank is the same as its number of rows. 
	 * 
	 * @return Returns true if matrix is full rank. Otherwise, returns false.
	 */
	 default boolean isFullRank() {	
		boolean result = false;
		Matrix A = (Matrix) this;
		
		if(A.rank() == A.m) {
			return true;
		}
		
		return result;
	}
	
	
	// TODO: javadoc
	 default boolean isSingular() {
		Matrix A = (Matrix) this;
		int tol = 13; // TODO: make this an optional parameter?
		
		return CNumber.round(A.det(), tol).equals(CNumber.ZERO);
	}
	
	
	/**
	 * Computes the 2-norm of a Matrix denoted L<sub>2, 2</sub> or A<sub>F</sub> for a matrix A. This is known as the 
	 * Frobenius norm or the Hilbert�Schmidt norm.
	 * 
	 * @returns Returns the Frobenius norm.
	 */
	// TODO: Look into other implementations Can be computed using square of singular values or with trace. 
	 default CNumber norm() {
		return norm(2, 2);
	}
	
	
	/**
	 * Computes the L<sub>p</sub> norm. This is equivalent to L<sub>p, q</sub>
	 * where p = q.
	 * <br>
	 * p can be <code>Double.POSITIVE_INFINITY</code>. which will result in
	 * the infinity / max norm being computed.
	 * 
	 * @param p - norm parameter
	 * @return
	 */
	 default CNumber norm(double p) {
		if(p == Double.POSITIVE_INFINITY) {
			return infNorm();
		}
		
		return norm(p, p);
	}
	
	
	/**
	 * Computes the L<sub>p, q</sub> norm of this matrix.
	 * 
	 * @param p - norm parameter 1
	 * @param q - norm parameter 2
	 * @return Returns L<sub>p, q</sub> norm of this matrix
	 */
	// TODO: Look into other implementations Can be computed using square of singular values or with trace. 
	// See https://en.wikipedia.org/wiki/Matrix_norm#L2,1_and_Lp,q_norms
	 default CNumber norm(double p, double q) {	
		double norm = 0;
		Matrix A = (Matrix) this;
		
		if(p < 1 || q < 1 || Double.isNaN(p) || p == Double.NEGATIVE_INFINITY) {
			throw new IllegalArgumentException("Expecting arguments to be greater than or equal to 1 but " +
					"got " + p + " and " + q + ".");
		}
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				norm += Math.pow(A.entries[i][j].mag(), p);
			}
			
			norm = Math.pow(norm, q/p);
		}
		
		return new CNumber(Math.pow(norm, 1/q));
	}
	
	
	 default CNumber infNorm() {
		CNumber[] values = LinAlgArrayUtils.flatten(((Matrix) this).entries);
		return new CNumber(CNumber.max(values).mag());
	}
	
	
	/**
	 * Finds an orthonormal basis for the row space of this matrix.
	 * <br><br>
	 * The row space of a matrix is the span of all column vectors. A orthonormal row space basis is 
	 * a linearly independent set of unit vectors which also spans the row space.
	 * 
	 * @return A matrix whose rows are row vectors forming an orthonormal basis for the row space of this matrix.
	 */
	 default Matrix rowSpace() {
		Matrix A = (Matrix) this,
			   rrefA = A.T().rref(),
			   C = new Matrix(rrefA.m, 0), // Orthonormal basis of column space
		       zero = new Vector(rrefA.m);
			
		/* Finds any columns which do not contain a pivot. Then indicates the variable is free.
		 * The number on non-pivot columns will be the number of eigenvectors for the associated eigenvalue.
		 */
		for(int j=0; j<rrefA.n; j++) { // Iterate over columns
			if(!rrefA.getColAsVector(j).equals(zero)) { // Then this column may have a pivot
				for(int i=rrefA.m-1; i>-1; i--) {
					if(!rrefA.entries[i][j].equals(CNumber.ZERO)) { // Then this may be a pivot
						if(rrefA.getSlice(i, i+1, 0, j).equals(Matrix.zeros(1, j))) { // Then rref[i][j] must be a pivot.
							C = C.augment(A.getColAsVector(j).scalDiv(A.getColAsVector(j).norm()));
						}
						
						break; // We can move to the next column.
					}
				}	
			}
		}
		
		return C.T();
	}
	
	
	/**
	 * Finds an orthonormal basis for the column space of this matrix.
	 * <br><br>
	 * The column space of a matrix is the span of all column vectors. A orthonormal column space basis is 
	 * a linearly independent set of unit vectors which also spans the column space.
	 * 
	 * @return A matrix whose columns are column vectors forming an orthonormal basis for the columns space of this matrix.
	 */
	 default Matrix colSpace() { // TODO: Return in matrix instead
		Matrix A = (Matrix) this,
			   rrefA = A.rref(),
			   C = new Matrix(rrefA.m, 0), // Orthonormal basis of column space
			   zero = new Vector(rrefA.m);
		
		/* Finds any columns which do not contain a pivot. Then indicates the variable is free.
		 * The number on non-pivot columns will be the number of eigenvectors for the associated eigenvalue.
		 */
		for(int j=0; j<rrefA.n; j++) { // Iterate over columns
			if(!rrefA.getColAsVector(j).equals(zero)) { // Then this column may have a pivot
				for(int i=rrefA.m-1; i>-1; i--) {
					if(!rrefA.entries[i][j].equals(CNumber.ZERO)) { // Then this may be a pivot
						if(rrefA.getSlice(i, i+1, 0, j).equals(Matrix.zeros(1, j))) { // Then rref[i][j] must be a pivot.
							C = C.augment(A.getColAsVector(j).scalDiv(A.getColAsVector(j).norm()));
						}
						
						break; // We can move to the next column.
					}
				}	
			}
		}
		
		
		
		
		
		return C;
	}
	
	
	/**
	 * Computes a orthonormal basis for the null space of this matrix. The null space of a matrix A is all vectors x that
	 * satisfy Ax = <bold>0</bold>.
	 * 
	 * @return A matrix whose column vectors from an orthonormal basis for the null space of this matrix.
	 */
	 default Matrix nullSpace() {
		Matrix A = ((Matrix) this).rref(),
			   x,
			   N = new Matrix(A.m, 0), // Matrix  containing orthonormal nullspace.
			   zero = new Vector(A.m); 
		ArrayList<Integer> nonpivCol = new ArrayList<Integer>();
		
		/* Finds any columns which do not contain a pivot. Then indicates the variable is free.
		 * The number on non-pivot columns will be the number of eigenvectors for the associated eigenvalue.
		 */
		for(int j=0; j<A.n; j++) { // columns
			if(A.getColAsVector(j).equals(zero)) { // Then this column does not have a pivot
				nonpivCol.add(j);
			} else {
				for(int i=A.m-1; i>-1; i--) {
					if(!A.entries[i][j].equals(CNumber.ZERO)) { // Then this may be a pivot
						if(!A.getSlice(i, i+1, 0, j).equals(Matrix.zeros(1, j))) { // Then rref[i][j] must not a pivot.
							nonpivCol.add(j); // Add this column to the list of non-pivot columns.
						}
						break; // We can move to the next column now.
					}
				}	
			}
		}
		
		
		for(int j : nonpivCol) {
			x = Matrix.zeros(A.m, 1);
			x.entries[j][0] = CNumber.ONE; // Set the free variable to one.
			
			for(int h=0; h<A.m; h++) {
				if(!nonpivCol.contains(h)) { // Ensure the variable is either not free.
					
					for(int k=0; k<A.m; k++) { // find the pivot row
						if(!A.entries[k][h].equals(CNumber.ZERO)) { // then we have found the row of the pivot.
							x.entries[h][0] = CNumber.addInv(A.entries[k][j]);
							break; // Move on to the next pivot column.
						}
					}
				}
			}
			
			N = N.augment(x.scalDiv(x.norm())); // Add x to the list of eigenvectors in V.
		}
		
		return N;
	}
	
	
	/**
	 * Computes an orthonormal basis of the left null space of this matrix. The left null space of a matrix A is all column vectors x
	 * which satisfy x<sup>T</sup>A=0<sup>T</sup>. This is equivalent to the the {@link #nullSpace() null space} of A<sup>T</sup>.
	 * 
	 * @return
	 */
	 default Matrix leftNullSpace() {
		return ((Matrix) this).T().nullSpace();
	}
	
	
	/**
	 * 
	 * @return True if matrix only contains positive real entries.
	 */
	 default boolean isPos() {
		Matrix A = (Matrix) this;
		boolean result = true;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(!(A.entries[i][j].isReal() && A.entries[i][j].compareToReal(CNumber.ZERO) > 0)) {
					result = false;
					return result; // Then we are done so we can return.
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * 
	 * @return True if matrix only contains negative entries.
	 */
	 default boolean isNeg() {
		Matrix A = (Matrix) this;
		boolean result = true;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(!(A.entries[i][j].isReal() && A.entries[i][j].compareToReal(CNumber.ZERO) < 0)) {
					result = false;
					return result; // Then we are done so we can return.
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * Checks if matrix is positive-definite. An m-by-m hermation (or symmetric if real) matrix M is positive-definite if for all non-zero
	 * column vectors z, z<sup>H</sup>Mz is positive where z<sup>H</sup> is the conjugate transpose of z.
	 * <br><br>
	 * Also see {@link #isPosSemidef() positive-semidefinite}.
	 * 
	 * @return True if the matrix is positive-definite. Otherwise, returns false.
	 */
	 default boolean isPosDef() {
		boolean result = false;
		
		
		if(this.isHermation()) {
			Matrix lam = this.eigVals();
			result = true;
			
			for(int i=0; i<lam.m; i++) {
				if(!(lam.entries[i][0].isReal() && lam.entries[i][0].compareToReal(CNumber.ZERO) > 0)) {
					// Then there is a negative or non-real eigenvalue, thus the matrix is not positive-definite
					result = false;
				}
			}
		} // Otherwise, the matrix is not positive-definite
		
		return result;
	}
	
	
	/**
	 * Checks if matrix is positive-semidefinite. An m-by-m hermation (or symmetric if real) matrix M is positive-definite if for all non-zero \
	 * column vector z, z<sup>*</sup>Mz is non-negative where z<sup>*</sup> is the conjugate transpose of z.
	 * <br><br>
	 * Also see {@link #isPosDef() positive-definite}.
	 * 
	 * @return True if the matrix is positive-semidefinite. Otherwise, returns false.
	 */
	 default boolean isPosSemidef() {
		boolean result = false;
		
		
		if(this.isHermation()) {
			Matrix lam = this.eigVals();
			result = true;
			
			for(int i=0; i<lam.m; i++) {
				if(!(lam.entries[i][0].isReal() && lam.entries[i][0].compareToReal(CNumber.ZERO) >= 0)) {
					// Then there is a negative or non-real eigenvalue, thus the matrix is not positive-semidefinite
					result = false;
				}
			}
		} // Otherwise, the matrix is not positive-semidefinite
		
		return result;
	}
	
	
	/**
	 * Computes the eigenvalues and associated eigenvectors for a square matrix.
	 * <br><br>
	 * Also see<br>
	 * - {@link #eigVecs() eigVecs()} to compute just the eigenvectors. <br>
	 * - {@link #eigVals() eigVals()} to compute just the eigenvalues. This is recommended if the eigenvectors are not needed as it will be faster.
	 * 
	 * @return Returns an array of two matrices. The first matrix is a row vector which contains the eigenvalues of A (no necessarily ordered but grouped by equality), repeated per there multiplicity.
	 *  The columns of the second matrix are the eigenvectors of A associated with each eigenvalue in the first matrix. For repeated eigenvalues, each associated eigenvector in the second matrix is 
	 *  an associated eigenvector.
	 */
	// TODO: add eigPairs which returns an array of matrices where each matrix is an eigenvector and the eigenvalue.
	 default Matrix[] eig() {
		Matrix[] eigenpairs = new Matrix[2];
		
		Matrix A = (Matrix) this;
		Matrix lam = Decompose.schur(A).diagAsVector().T(); // Get eigenvalues of A using the schur decomposition
		lam = new Vector(LinAlgArrayUtils.group(lam.entries[0])).round(13); // Round eigenvalues to near machine epsilon and group equivalent eigenvalues.
		
		Matrix I = Matrix.I(A.m),
				   zero = new Matrix(A.m, 1),
				   x = new Matrix(A.m, 1), // Stores a specific eigenvector
				   V = new Matrix(A.m, 0); // Stores eigenvectors in its columns
			
		Vector b = new Vector(A.m);
		ArrayList<Integer> nonpivCol = new ArrayList<Integer>();	

		for(int h=0; h<lam.m; h++) {
			
			if(h==0 || !lam.entries[h][0].equals(lam.entries[h-1][0])) { // Only compute eigenvectors for a given eigenvalue once.
				Matrix rref = A.sub(I.scalMult(lam.entries[h][0]));
				
				/* Compute null space of (A-nI) where n is the specified eigenvalue.
				 *  This is equivalent to the eigenspace of A for the eigenvalue n.
				 */
				V = V.augment(rref.nullSpace());
			}
		}
		
		eigenpairs[0] = lam;
		eigenpairs[1] = V;
		
		return eigenpairs;
	}
	
	
	/**
	 * Computes the right eigenvectors of a matrix. This is done by first computing the Schur decomposition.
	 * <br><br>
	 * Also see<br>
	 * - {@link #eig() eig()} to compute both eigenvalues and eigenvectors.<br>
	 * - {@link #eigVals() eigVals()} to compute just the eigenvalues.
	 * 
	 * @return Returns a matrix containing the eigenvectors of this matrix as its column vectors.
	 */
	default Matrix eigVecs() {
		return this.eig()[1];
	}
	
	
	/**
	 * Computes the eigenvalues of a matrix. This is done by first computing the Schur decomposition.
	 * <br><br>
	 * Also see<br>
	 * - {@link #eig() eig()} to compute both eigenvalues and eigenvectors.<br>
	 * - {@link #eigVecs() eigVecs()} to compute just the eigenvectors.
	 * 
	 * @return Returns a column vector containing the eigenvalues of this matrix.
	 */
	 default Matrix eigVals() {
		Matrix A = (Matrix) this;
		return Decompose.schur(A).diagAsVector();
	}
	
	
	/**
	 * Checks if a matrix is diagonalizable. A matrix B is diagonalizable if and only if
	 * the multiplicity for each eigenvalue is equivalent to the eigenspace for that eigenvalue.
	 * 
	 * @return True if the matrix is diagonalizable. Otherwise, returns false.
	 */
	 default boolean isDiagonalizable() {
		Matrix A = (Matrix) this,
			   I = Matrix.I(A.m); 
		boolean result = true;
		int count = 0;
		
		Matrix lam = new Vector(LinAlgArrayUtils.group(A.eigVals().T().entries[0])),
			   nullSpace;
		ArrayList<Integer> multiplicites = new ArrayList<Integer>(); // Contains multiplicity for each eigenvalue
		ArrayList<Integer> eigenDim = new ArrayList<Integer>(); // Contains dimension of each eigenspace for all eigenvalues.
		
		for(int i=0; i<lam.m; i++) { // Find multiplicity for each eigenvalue
			if((i!=0 && !lam.entries[i][0].equals(lam.entries[i-1][0])) || i==0) {
				count=0;
				
				for(int j=0; j<lam.m; j++) {
					if(lam.entries[j][0].equals(lam.entries[i][0])) {
						count++;
					}
				}
				
				multiplicites.add(count);
				
				nullSpace = A.sub(I.scalMult(lam.entries[i][0])).nullSpace();
				eigenDim.add(nullSpace.n);
			}
		}
		
		for(int i=0; i<eigenDim.size(); i++) {
			if(eigenDim.get(i).equals(multiplicites.get(i))) {
				result = false;
				break;
			}
		}
		
		return result;
	}
	
	
	 static void main(String args[]) {
		int[][] a = {{0, 1, 0, 0},
					 {0, 0, 1, 0},
					 {0, 0, 0, 1},
					 {1, 0, 0, 0}}; // Schur decomposition will fail here because we have 4 eigenvalues all with magnitude 1.
		
		double[][] 	a1 =  {	{2, 5, 8, 7},
		  					{5, 2, 2, 8},
		  					{7, 5, 6, 6},
		  					{5, 4, 4, 8} };
		
		double[][] a2 = {{1, -3, 3},
		                 {3, -5, 3},
		                 {6, -6, 4}};
		
		double[][] a3 = {{0,  0,  0, 1},
						 {0,  0, -1, 0},
						 {0,  1,  0, 0},
						 {-1, 0, 0, 0}}; // TODO: Currently fails possibly due to repeated eigenvalues
		
		CNumber[][] ac = {	{new CNumber("2+2i"),	new CNumber("5"), new CNumber("8"), new CNumber("7")},
							{new CNumber("5"), 		new CNumber("i"), new CNumber("2"), new CNumber("8")},
							{new CNumber("7"),		new CNumber("5"), new CNumber("6"), new CNumber("6")},
							{new CNumber("5"), 		new CNumber("4"), new CNumber("4"), new CNumber("8")} }; // TODO: Currently fails possibly due to complex conjugate eigenvalues
		
		CNumber[][] ac2 = {	{new CNumber("2+2i"),	new CNumber("5"), new CNumber("8")},
							{new CNumber("5"), 		new CNumber("i"), new CNumber("2")},
							{new CNumber("7"),		new CNumber("5"), new CNumber("6")}};
		
		double[][] test1 = {{1, 0,  1},
							{0, 1, -2},
							{0, 0,  0}};
		
		double[][] test2 = {{0, 1, 0, 2},
							{0, 0, 1, 2},
							{0, 0, 0, 0},
							{0, 0, 0, 0}};
		
		double[][] test3 = {{1, 0, -1, 0},
							{0, 1, -1, 0},
							{0, 0,  0, 1},
							{0, 0,  0, 0}};
		
		
		Matrix A = new Matrix(test3);
		

		Matrix[] VW = A.eig();
		
		Matrix.print("A:\n", A, "\n\n");
		Matrix.print("lam:\n", VW[0], "\n\n");
		Matrix.print("V:\n", VW[1], "\n\n");
		
		for(int i=0; i<VW[1].n ; i++) { // Ensure vectors are actually eigenvectors.
			System.out.println(A.mult(VW[1].getColAsVector(i)).round(10).equals(VW[1].getColAsVector(i).scalMult(VW[0].entries[i][0]).round(10)));
		}
		
		
		A.isDiagonalizable();

	}
}
