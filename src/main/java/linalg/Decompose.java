
package linalg;

/**
 * This class provided support for several decompositions of real or complex matrices.
 * <br><br>
 * Supported Decompositions:<br>
 * - {@link #LU(Matrix) LU Decomposition}<br>
 * - {@link #LUP(Matrix) LUP Decomposition}<br>
 * - {@link #LUPQ(Matrix) LUPQ Decomposition}<br>
 * - {@link #LDU(Matrix) LDU Decomposition}<br>
 * - {@link #QR(Matrix) QR Decomposition}<br>
 * - {@link #cholesky(Matrix) Cholesky Decomposition}<br>
 * - {@link #schur(Matrix) Complex Schur Decomposition}<br>
 * - {@link #schurReal(Matrix) Real Schur Decomposition}<br>
 * - {@link #schur(Matrix, boolean) Optional Real or Complex Schur Decomposition}<br>
 * 
 *	// TODO: Add Hessenburg decomposition. i.e. A=PHP*
 * 
 * @author Jacob Watters
 */
public class Decompose {
	
	// TODO: All LU factorizations should generalize to rectangular matrices.
	// TODO: The cholesky factorization do not function properly for complex matrices.
	
	/**
	 * Factors a square matrix into a lower triangular matrix, L, and an upper triangular matrix, U, such that A = L*U. <br>
	 * Note: This method does not use partial pivoting. To use partial pivoting see one of the following <br><br>
	 *  - {@link #LUP(Matrix) LUP(Matrix A)} <br>
	 *  - {@link #LUPQ(Matrix) LUPQ(Matrix A)}<br>
	 *  
	 *  Also see {@link #LDU(Matrix) LDU(Matrix A)}
	 * 
	 * @param A - Matrix to decompose.
	 * @return An array of matrices representing LU decomposition using NO pivoting. The array contains in order [L, U]
	 */
	public static Matrix[] LU(Matrix A) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		return LUDecomposition.exicute(A, LUDecomposition.LU); // This will be two Matrices (L and U);
	}
	
	
	/**
	 * Factors a square matrix into a lower triangular matrix, L, and an upper triangular matrix, U using row swaps represented
	 * by a Matrix pivot matrix P such that PA = LU where P is the permutation matrix describing row swaps in A during
	 * partial pivoting.
	 * <br><br>
	 * Also see<br>
	 * 	- {@link #LU(Matrix) LU(Matrix A)}<br>
	 *  - {@link #LUPQ(Matrix) LUPQ(Matrix A)}<br>
	 *  - {@link #LDU(Matrix) LDU(Matrix A)}
	 * 
	 * @param A - Matrix to decompose.
	 * @return An array of matrices representing LU decomposition using partial pivoting. The array contains in order [L, U, P]
	 */
	public static Matrix[] LUP(Matrix A) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		return LUDecomposition.exicute(A, LUDecomposition.LUP);
	}
	
	
	/**
	 * Factors a square matrix A into a lower triangular matrix, L, and an upper triangular matrix, U using row and column swaps represented
	 * by the permutation matrices P and Q such that PAQ = LU where P is the permutation matrix describing row swaps in A and Q is the permutation matrix
	 * describing column swaps in A.
	 * 
	 *<br><br>
	 * Also see<br>
	 * 	- {@link #LU(Matrix) LU(Matrix A)} for LU decomposition with NO pivoting<br>
	 *  - {@link #LUP(Matrix) LUP(Matrix A)} for LU decomposition using partial pivoting.<br>
	 *  - {@link #LDU(Matrix) LDU(Matrix A)}
	 * 
	 * @param A - Matrix to decompose.
	 * @return An array of matrices representing LU decomposition using full pivoting. The array contains in order [L, U, P, Q]
	 */
	public static Matrix[] LUPQ(Matrix A) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		return LUDecomposition.exicute(A, LUDecomposition.LUPQ);
	}
	
	
	/**
	 * Decomposes a square matrix A into a Lower triangular matrix L, a diagonal matrix D, and an upper triangular matrix U such that A = LDU.  
	 * 
	 * 	- {@link #LU(Matrix) LU(Matrix A)} for LU decomposition with NO pivoting
	 *  - {@link #LUP(Matrix) LUP(Matrix A)} for LU decomposition using partial pivoting.
	 *  - {@link #LUPQ(Matrix) LUPQ(Matrix A)} for LU decomposition using full pivoting
	 * 
	 * @param A - Matrix to decompose.
	 * @return Returns an array of matrices containing in order, L, D, and U.
	 */
	public static Matrix[] LDU(Matrix A) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got " + A.shape);
		}
		
		return LUDecomposition.exicute(A, LUDecomposition.LDU);
	}
	
	
	/**
	 * Computes the QR decomposition of a matrix A such that A=QR, Q is an orthogonal matrix and R is an upper triangular matrix. 
	 * <br><br>
	 * If you would like to know the number of householder reflectors used in the QR 
	 * decomposition, see {@link #QR(Matrix, boolean) QR(Matrix A, boolean countHouseholder)} 
	 * 
	 * @param A - a rectangular matrix with linearly independent columns.
	 * @return An array of matrices of length 2 containing, in order, Q and R.
	 */
	public static Matrix[] QR(Matrix A) {
		return QRDecomposition.exicute(A, QRDecomposition.QR_NORMAL); 
	}
	
	
	/**
	 * Computes the QR decomposition of a matrix A such that A=QR, Q is an orthogonal matrix and R is an upper triangular matrix.
	 * <br><br>
	 * If you are not interested in knowing the number of householder reflectors used in the QR 
	 * decomposition, see {@link #QR(Matrix) QR(Matrix A)} 
	 * 
	 * @param A - a rectangular matrix with linearly independent columns.
	 * @param countHouseholder
	 * @return An array of matrices of length 2 or 3. The first and second matrix are always Q and R respectively. If countHouseholder flag is true, then
	 * a third 1-by-1 matrix containing the number of Householder reflectors used will also be returned.
	 */
	public static Matrix[] QR(Matrix A, boolean countHouseholder) {
		return QRDecomposition.exicute(A, QRDecomposition.QR_COUNT_HOUSEHOLDER); 
	}
	
	
	/**
	 * Computes the Cholesky decomposition of a Hermation, positive-definite matrix into the product 
	 * of a lower triangular matrix and its conjugate transpose. This is similar to the 
	 * {@link LU(Matrix) LU decomposition} with the special requirement that L=U<sup>*</sup> where U<sup>*</sup> is 
	 * the conjugate transpose of U.
	 * 
	 * @param A - A hermation, positive-definite matrix to be decomposed.
	 * @return A matrix array containing in order a lower triangular matrix and its conjugate transpose. 
	 */
	public static Matrix[] cholesky(Matrix A) {
		if(!A.isPosDef()) {
			throw new IllegalArgumentException("Matrix must be positive-definite.");
		}
		
		return CholeskyDecomposition.choleskyDecomp(A);
	}
	
	
	/**
	 * Computes the complex Schur decomposition of this matrix. That is decomposes a matrix A such that A = QTQ<sup>*</sup>
	 * where Q is a unitary matrix (i.e. Q*Q=I), and T is an upper triangular matrix which contains the corresponding eigenvalues of A
	 * along its diagonal.
	 * <br><br>
	 * Also see {@link #schurReal(Matrix) schurReal(Matrix A)} which defaults to the real Schur decomposition and 
	 * {@link #schur(Matrix, boolean) schur(Matrix A, boolean complex)} for optional real or complex Schur decomposition.
	 * 
	 * @param A - Matrix to decompose
	 * @return An array of matrices of length three containing in order [Q, E, Q<sup>-1</sup>].
	 */
	public static Matrix schur(Matrix A) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got shape " + A.shape);
		}
		
		return SchurDecomposition.schurDecomp(A, true, true);
	}
	
	
	/**
	 * Computes the real or complex Schur decomposition of this matrix. That is decomposes a matrix A such that A = QTQ<sup>*</sup>
	 * where Q is a unitary matrix (i.e. Q*Q=I). In the complex Schur decomposition, T is an upper triangular matrix which contains the corresponding eigenvalues of A
	 * along its diagonal. In the real Schur decomposition, T is a block upper diagonal matrix containing real eigenvalues of A along the diagonal and representing
	 * any complex conjugate pair eigenvalues as a real 2-by-2 matrix. The eigenvalues of this 2-by-2 matrix are the complex conjugate eigenvalues of A.
	 * <br><br>
	 * Also see {@link #schur(Matrix) schur(Matrix A)} which defaults to the complex Schur decomposition and 
	 * {@link #schurReal(Matrix) schurReal(Matrix A)} which defaults to the real Schur decomposition.
	 * 
	 * @param A - Matrix to decompose
	 * @param complex - Flag to compute real or complex Schur decomposition. If true, the complex Schur decomposition will be computed. If false
	 * 		the real Schur decomposition will be computed.
	 * @return An array of matrices of length two containing in order [Q, T]. // TODO: Change schurDecomp so this is actually what is returned
	 */
	public static Matrix schur(Matrix A, boolean complex) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got shape " + A.shape);
		}
		
		return SchurDecomposition.schurDecomp(A, complex, true);
	}
	
	
	/**
	 * Computes the real Schur decomposition of this matrix. That is decomposes a matrix A such that A = QTQ<sup>*</sup>
	 * where Q is a unitary matrix (i.e. Q<sup>*</sup>Q=I) and T is a block upper diagonal matrix containing real eigenvalues of A along the diagonal and representing
	 * any complex conjugate pair eigenvalues as a real 2-by-2 matrix. The eigenvalues of this 2-by-2 matrix are the complex conjugate eigenvalues of A.
	 * <br><br>
	 * Also see {@link #schur(Matrix) schur(Matrix A)} which defaults to the complex Schur decomposition and 
	 * {@link #schur(Matrix, boolean) schur(Matrix A, boolean complex)} for optional real or complex Schur decomposition.
	 * 
	 * @param A - Matrix to decompose
	 * @param complex - Flag to compute real or complex Schur decomposition. If true, the complex Schur decomposition will be computed. If false
	 * 		the real Schur decomposition will be computed.
	 * @return An array of matrices of length two containing in order [Q, T]. // TODO: Change schurDecomp so this is actually what is returned
	 */
	public static Matrix schurReal(Matrix A) {
		if(!A.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got shape " + A.shape);
		}
		
		return SchurDecomposition.schurDecomp(A, false, true);
	}
}
