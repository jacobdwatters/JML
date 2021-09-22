package com.jml.linalg;

import com.jml.linalg.complex_number.CNumber;


class SchurDecomposition {
	
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
}
