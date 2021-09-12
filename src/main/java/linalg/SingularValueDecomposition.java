package linalg;

class SingularValueDecomposition {
	
	
	/**
	 * Computes the singular value decomposition of an m-by n matrix.
	 * <br><br>
	 * Warning, the way this method is currently implemented is slow and numerically unstable. 
	 * For a matrix with a large condition number, digits of accuracy may be lost.
	 * 
	 * @param A - An m-by-n matrix to compute singular value decomposition of.
	 * @return The singular value decomposition 
	 */
	protected static Matrix[] svd(Matrix A) {
		Matrix[] result = new Matrix[3];
		
		Matrix B = A.H().mult(A).round(13);
		
		Matrix U = A.mult(A.H()).eigVecs();
		Matrix[] SV = B.eig(); // Get square of singular values and right singular vectors of A.
		
		SV[0] = MatrixManipulations.toDiag(SV[0].sqrt().T().entries[0]);
		SV[0] = SV[0].stack(Matrix.zeros(A.m-SV[0].m));
		
		result[0] = U;
		result[1] = SV[0];
		result[2] = SV[1].H();
		
		
		Matrix.print("B:\n", B, "\n\n");
		Matrix.print("U:\n", U, "\n\n");
		Matrix.print("S:\n", SV[0], "\n\n");
		Matrix.print("V:\n", SV[1], "\n\n");
		Matrix.print("Result:\n", U.mult(SV[0]).mult(SV[1].H()), "\n\n");
		
		return result;
	}
	
	
	
	public static void main(String[] args) {
		double[][] a = {{-4,   0.8,  2.6, 0.6},
						{-4.5, 1.9, -0.7, 3.3}}; 	// Schur decomposition fails for this  B = [[0, A^T]
													//											[A, 0  ]]
		double[][] b = {{1, 2, 3, 4},
						{2, 1, 5, 6},
						{3, 5, 1, 7},
						{4, 6, 7, 1}};
		
		double[][] c = {{ 3,  3},
						{-3, -3},
						{-1,  1},
						{ 1, -1}};
		
		Matrix A = new Matrix(c);
		
		Matrix.print("A:\n", A, "\n\n");

		svd(A);
	}
	
}
