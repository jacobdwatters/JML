package linalg;

import linalg.complex_number.CNumber;

public class Solvers {

	
	/**
	 * Solves multiple real or complex systems of equations with the same coefficient matrix. That is, solves for the vector x in
	 * Ax=b for many different b vectors.
	 * 
	 * @param A - Coefficient matrix for system.
	 * @param b - List of constant vectors. If a row vector is passed, it will be implicitly converted to a column vector.
	 * @return
	 */
	public static Vector[] solve(Matrix A, Vector...b) {	
		if(A.isSingular()) {
			throw new IllegalArgumentException("Matrix is singular.");
		}
		
		Vector y[] = new Vector[b.length];
		Vector x[] = new Vector[b.length];

		Matrix LU[] = Decompose.LUPQ(A);
		
		for(int i=0; i<b.length; i++) {
			y[i] = forwardSolve(LU[0], LU[2].mult(b[i].toColVector()).toVector());
			x[i] = backSolve(LU[1], y[i]);
			
			x[i] = LU[3].mult(x[i]).toVector();
		}
		
		return x;
	}
	

	/**
	 * Solves a real or complex linear System of equations of from Ax=b for the vector x in an exact sense.
	 * Here A is the coefficient matrix and b is a vector of constants. This is done using the LUPQ factorization of A.
	 *
	 * @param A - Coefficient Matrix for system.
	 * @param b - Vector of constants. If a row vector is passed, it will be implicitly converted to a column vector.
	 * @return a column vector x that is the result of solving Ax=b.
	 */
	public static Vector solve(Matrix A, Vector b) {
		if(A.isSingular()) { // TODO: A.det() fails for some matrices meaning the isSingular method does not work
			throw new IllegalArgumentException("Matrix is singular.");
		}
		
		Matrix LU[] = Decompose.LUPQ(A);

		Vector y = forwardSolve(LU[0], LU[2].mult(b.toColVector()).toVector());
		Vector x = backSolve(LU[1], y);

		return LU[3].mult(x).toVector();
	}


	/**
	 * Solves a real or complex linear System of equations of from Ax=b for the vector x in an exact sense.
	 * Here A is the coefficient matrix and b is a vector of constants. This is done using the LUPQ factorization of A.
	 *
	 * @param A - Coefficient Matrix for system.
	 * @param b - Vector of constants. If a row vector is passed, it will be implicitly converted to a column vector.
	 * @return a column vector x that is the result of solving Ax=b.
	 */
	public static Vector sovle(Matrix A, Matrix b) {
		if(A.isSingular()) {
			throw new IllegalArgumentException("Matrix is singular.");
		}
		
		return solve(A, b.toVector());
	}


	/**
	 * Solves a real or complex linear System of equations of from Ax=b for the vector x in an exact sense.
	 * Here A is the coefficient matrix and b is a vector of constants. This is done using the QR factorization of A.
	 *
	 * @param A - coefficient Matrix for system.
	 * @param b - Vector of constants. If a row vector is passed, it will be implicitly converted to a column vector.
	 * @return a column vector x that is the result of solving Ax=b.
	 */
	public static Vector solveQR(Matrix A, Vector b) {
		if(A.isSingular()) {
			throw new IllegalArgumentException("Matrix is singular.");
		}
		
		Matrix[] QR = Decompose.QR(A);
		Vector b_hat = QR[0].H().mult(b.toColVector()).toVector();

		// Now the equivalent system Rx = b_hat must be solved. Because R is upper triangular, this can be done with back substitution.
		return backSolve(QR[1], b_hat);
	}


	/**
	 * Solves a real or complex linear System of equations of from Ax=b for the vector x in an exact sense.
	 * Here A is the coefficient matrix and b is a vector of constants. This is done using the QR factorization of A.
	 *
	 * @param A - Coefficient Matrix for system.
	 * @param b - Vector of constants. If a row vector is passed, it will be implicitly converted to a column vector.
	 * @return a column vector x that is the result of solving Ax=b.
	 */
	public static Vector solveQR(Matrix A, Matrix b) {
		if(A.isSingular()) {
			throw new IllegalArgumentException("Matrix is singular.");
		}
		
		return solveQR(A, b.toVector());
	}


	/**
	 * This is a helper method to aid in solving systems of equations.
	 * It performs back substitution for an upper triangular matrix R and a vector b_hat.
	 * That is, this method solves the system Rx=b.
	 *
	 * @param b - Vector of constants
	 * @return Returns the result of back substitution on the linear System.
	 */
	protected static Vector backSolve(Matrix R, Vector b) {
		int m = R.m;
		CNumber xSum = CNumber.ZERO;
		Vector x = new Vector(m, Vector.COLUMN_VECTOR);

		x.entries[m-1][0] = CNumber.divide(b.entries[m-1][0], R.entries[m-1][m-1]);

		for(int i=m-1; i>-1; i--) {
			xSum = CNumber.ZERO;

			for(int j=i+1; j<m; j++) {
				xSum = CNumber.add(
							xSum,
							CNumber.multiply(
									R.entries[i][j],
									x.entries[j][0]
							)
						);
			}

			x.entries[i][0] = CNumber.divide(
								CNumber.subtract(
										b.entries[i][0],
										xSum
								),
								R.entries[i][i]
						   );
		}


		return x;
	}


	/**
	 * This is a helper method to aid in solving systems of equations.
	 * It performs forward substitution for a lower triangular matrix L and a vector b.
	 * That is, this method solves the system Lx=b.
	 *
	 * @param L - Lower triangular matrix
	 * @param b - Vector of constants
	 * @return Returns the result of forward substitution on the linear System.
	 */
	protected static Vector forwardSolve(Matrix L, Vector b) {
		int m = L.m;
		CNumber ySum = CNumber.ZERO;
		Vector y = new Vector(m, Vector.COLUMN_VECTOR);

		y.entries[0][0] = b.entries[0][0];

		for(int i=0; i<L.m; i++) {
			ySum = CNumber.ZERO;

			for(int j=i-1; j>-1; j--) {
				ySum = CNumber.add(
					ySum,
					CNumber.multiply(
						L.entries[i][j],
						y.entries[j][0]
					)
				);
			}

			y.entries[i][0] = CNumber.subtract(
									b.entries[i][0],
									ySum
							  );
		}

		return y;
	}


	// FOR DEVELOPMENT TESTING ONLY //
	public static void main(String args[]) {
		double[][] A_entries = {{1, -2, 3},
								{2, 1, 1},
								{-3, 2, -2}};

		double[] b1_entries = {7, 4, -10};
		double[] b2_entries  = {2, 3, 5};
		double[] b3_entries  = {4, 1, 90};

		Matrix A = new Matrix(A_entries);
		
		Vector b1 = new Vector(b1_entries);
		Vector b2 = new Vector(b2_entries, Vector.ROW_VECTOR);
		Vector b3 = new Vector(b3_entries);
		
		Vector x = solveQR(A, b1);
		Vector[] x2 = solve(A, b1, b2, b3);

		Matrix.print("A:\n", A, "\n\n");
		Matrix.print("b:\n", b1, "\n\n");
		Matrix.print("x:\n", x.round(10), "\n\n");
		Matrix.print("x1:\n", x2[0].round(10), "\n\n");
		Matrix.print("x2:\n", x2[1].round(10), "\n\n");
		Matrix.print("x3:\n", x2[2].round(10), "\n\n");
	}
}
