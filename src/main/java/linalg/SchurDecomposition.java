package linalg;

import linalg.complex_number.CNumber;


public class SchurDecomposition {
	/**
	 * Computes the Schur decomposition of this matrix. That is decomposes a symmetric matrix A such that A = QRQ<sup>-1</sup>
	 * where Q is an orthogonal matrix thus Q<sup>-1</sup>=Q<sup>*</sup> where Q<sup>*</sup> is the conjugate transpose of Q, and R is an upper-triangular matrix which contains the eigenvalues of A
	 * along its diagonal.
	 * 
	 * @param A - Matrix to decompose
	 * @return An array of matrices of length two containing in order [Q, E].
	 */
	// TODO: The QR algorithm here needs to be improved to be more general.
	protected static Matrix schurTest(Matrix A, boolean complex, boolean computeQ) {
		Matrix[] HV = A.hessuV(); // Convert the matrix to upper Hessenberg form and return reflectors used.
		Matrix H = HV[0], V = HV[1];
		
		Matrix.print("H:\n", H, "\n\n");
		Matrix.print("V:\n", V, "\n\n");
		
		double tol = 1E-14; // TODO: Add overloaded method that allows tol and countTol to be passed
		int countTol = 800, 
			count,
			n = A.m-1;
		
		Matrix lam = Matrix.zeros(A.m, 1), // Stores eigenvalues of A. 
			   QR[] = null,
			   T = Matrix.I(A.m),
			   Qbar = Matrix.I(A.m),
			   Q = Matrix.I(A.m);
		
		CNumber mu = null, // mu is the shift for the shifted QR Algorithm. Currently this shift is computed using Rayleigh quotient. // TODO: Change to Wilkinson shift
				disc, 
				two = new CNumber(2); 
		
		while(n>0) {
			count = 0;
			
			while(H.getSlice(n, n+1, 0, n).abs().max().re > tol && count<countTol) {
				count++;
				mu = H.entries[n][n];
				
				QR = Decompose.QR(H.sub(Matrix.I(n+1).scalMult(mu)).round(14)); // Compute the QR factorization of A with a shift
				H = QR[1].mult(QR[0]).add(Matrix.I(n+1).scalMult(mu).round(14)); // Reverse the shift.
			}

			T.setSlice(H.m-(n+1), H.m-(n+1), H);
			
			if(count<countTol) { // Then we have an isolated 1-by-1 block
				lam.entries[n][0] = H.entries[n][n]; // Store eigenvalue
				n--;
				H = H.getSlice(0, n+1, 0, n+1); // Deflate by one
			} 
			else { // Then we have an isolated 2-by-2 block
				
				// Computes the complex eigenvalues from the 2-by-2 block matrix. 	// TODO: This really only needs to be done
																					// when complex==true. Otherwise, we can leave as is.

				
				if(complex) { // Then convert to the complex schur form.
					disc = 	CNumber.add(
							CNumber.pow(
								CNumber.subtract(H.entries[n-1][n-1], H.entries[n][n]),
								2
							),
							CNumber.multiply(
									new CNumber(4), 
									CNumber.multiply(H.entries[n][n-1], H.entries[n-1][n])
							)
						
						);
					lam.entries[n][0] = CNumber.divide(
											CNumber.add(
													CNumber.add(H.entries[n-1][n-1], H.entries[n][n]),
													CNumber.sqrt(disc)
											),
											two
										);
					
					lam.entries[n-1][0] = CNumber.divide(
											CNumber.subtract(
													CNumber.add(H.entries[n-1][n-1], H.entries[n][n]),
													CNumber.sqrt(disc)
											),
											two
										  );
					
					Matrix zero = Matrix.zeros(2);
					T.setSlice(n-1, n-1, zero);
					
					T.entries[n][n] = lam.entries[n][0];
					T.entries[n-1][n-1] = lam.entries[n-1][0];
				}
				
				n=n-2;
				H = H.getSlice(0, n+1, 0, n+1); // Deflate by 2
			}
		}
		
		if(n>-1) { // Then only a 1-by-1 block remains
			lam.entries[0][0] = H.entries[0][0];
		}
		
		T.roundToZero(13);
		
		return T; // TODO: Should really return T and Q. Still need to actually compute Q
	}
	
	
	// TODO: T has been successfully calculated for the Schur decomposition H=QTQ*. We still need to figure out how to compute Q.
	protected static Matrix schurDecomp(Matrix A, boolean complex, boolean computeQ) {
		Matrix H = A.hessu(); // Convert the matrix to upper Hessenberg form.
		
		double tol = 1E-14; // TODO: Add overloaded method that allows tol and countTol to be passed
		int countTol = 800, 
			count,
			n = A.m-1;
		
		Matrix lam = Matrix.zeros(A.m, 1), // Stores eigenvalues of A. 
			   QR[] = null,
			   T = Matrix.I(A.m),
			   Qbar = Matrix.I(A.m),
			   Q = Matrix.I(A.m);
		
		CNumber mu = null, // mu is the shift for the shifted QR Algorithm. Currently this shift is computed using Rayleigh quotient. // TODO: Change to Wilkinson shift
				disc, 
				two = new CNumber(2); 
		
		while(n>0) {
			count = 0;
			
			while(H.getSlice(n, n+1, 0, n).abs().max().re > tol && count<countTol) {
				count++;
				mu = H.entries[n][n];
				
				QR = Decompose.QR(H.sub(Matrix.I(n+1).scalMult(mu)).round(14)); // Compute the QR factorization of A with a shift
				H = QR[1].mult(QR[0]).add(Matrix.I(n+1).scalMult(mu).round(14)); // Reverse the shift.
			}

			T.setSlice(H.m-(n+1), H.m-(n+1), H);
			
			if(count<countTol) { // Then we have an isolated 1-by-1 block
				lam.entries[n][0] = H.entries[n][n]; // Store eigenvalue
				n--;
				H = H.getSlice(0, n+1, 0, n+1); // Deflate by one
			} 
			else { // Then we have an isolated 2-by-2 block
				
				// Computes the complex eigenvalues from the 2-by-2 block matrix. 	// TODO: This really only needs to be done
																					// when complex==true. Otherwise, we can leave as is.

				
				if(complex) { // Then convert to the complex schur form.
					disc = 	CNumber.add(
							CNumber.pow(
								CNumber.subtract(H.entries[n-1][n-1], H.entries[n][n]),
								2
							),
							CNumber.multiply(
									new CNumber(4), 
									CNumber.multiply(H.entries[n][n-1], H.entries[n-1][n])
							)
						
						);
					lam.entries[n][0] = CNumber.divide(
											CNumber.add(
													CNumber.add(H.entries[n-1][n-1], H.entries[n][n]),
													CNumber.sqrt(disc)
											),
											two
										);
					
					lam.entries[n-1][0] = CNumber.divide(
											CNumber.subtract(
													CNumber.add(H.entries[n-1][n-1], H.entries[n][n]),
													CNumber.sqrt(disc)
											),
											two
										  );
					
					Matrix zero = Matrix.zeros(2);
					T.setSlice(n-1, n-1, zero);
					
					T.entries[n][n] = lam.entries[n][0];
					T.entries[n-1][n-1] = lam.entries[n-1][0];
				}
				
				n=n-2;
				H = H.getSlice(0, n+1, 0, n+1); // Deflate by 2
			}
		}
		
		if(n>-1) { // Then only a 1-by-1 block remains
			lam.entries[0][0] = H.entries[0][0];
		}
		
		T.roundToZero(13);
		
		return T; // TODO: Should really return T and Q. Still need to actually compute Q
	}
	
	
	public static void main(String[] args) {
		double[][] 	a =  {	{2, 5, 8, 7},
		 		  			{5, 2, 2, 8},
		 		  			{7, 5, 6, 6},
		 		  			{5, 4, 4, 8} };
		double[][] a2 = {{0,  0,  0, 1},
						 {0,  0, -1, 0},
						 {0,  1,  0, 0},
						 {-1, 0, 0, 0}};
		
		double[][] test = { {1, 0, 0},
							{0, 1, 2},
							{0, 0, 0}};
		
		double[][] test2 = {{0, 1},
							{1, 0}};
		double[][] b = {{1, 2, 3, 4},
						{2, 2, 5, 6},
						{3, 5, 3, 7},
						{4, 6, 7, 4}};
		
		Matrix A = new Matrix(b);
		
		Matrix lam = schurTest(A, true, false);
		
		Matrix.print("-------------------------------------------------------------");
		Matrix.print("A:\n", A, "\n\n");
		Matrix.print("lam:\n", lam, "\n\n");
	}
}
