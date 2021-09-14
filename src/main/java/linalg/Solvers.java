package linalg;

import linalg.complex_number.CNumber;

public class Solvers {

	private static final String IS_SINGULAR_ERR = "Matrix is singular.";

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
			throw new IllegalArgumentException(IS_SINGULAR_ERR);
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
			throw new IllegalArgumentException(IS_SINGULAR_ERR);
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
			throw new IllegalArgumentException(IS_SINGULAR_ERR);
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
			throw new IllegalArgumentException(IS_SINGULAR_ERR);
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
			throw new IllegalArgumentException(IS_SINGULAR_ERR);
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
}
