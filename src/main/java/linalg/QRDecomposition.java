package linalg;

import linalg.complex_number.CNumber;

/**
 * An interface providing functionality for QR factorization.
 * 
 * @author Jacob Watters
 */
class QRDecomposition {
	
	final static int QR_NORMAL = 0;
	final static int QR_COUNT_HOUSEHOLDER = 1;
	
	protected static Matrix[] exicute(Matrix A, int type) {
		return QR(A, type == QR_COUNT_HOUSEHOLDER);
	}
	
	/**
	 * Factors a matrix <code>A</code> into an orthogonal matrix <code>Q</code> and an upper
	 * triangular matrix <code>R</code> such that <code>A=QR</code>.
	 * 
	 * @param A - Matrix to factor into QR. 
	 * @return QR factorization of the matrix A as an array of matrices of length 2. 
	 * The first element of the array will be Q and the second will be R.
	 */
	protected static Matrix[] QR(Matrix A, boolean countHouseholder) {
		Matrix QR[], 
				Q = Matrix.I(A.m), 
				R = A,
				H,
				v, x; // Column Vector
		
		Vector w;
		CNumber norm, two = new CNumber(2);
		CNumber n = CNumber.ZERO;
		int counth = 0; // Counts the number of householder Reflectors used.
		
		for(int j=0; j<A.n; j++) {
			
			if(!R.getSlice(j, A.m, j, j+1).equals(new Matrix(A.m-j, 1))) {

				x = R.getSlice(j, A.m, j, j+1);		
				norm = x.norm();
				
				w = new Vector(A.m-j);
				w.entries[0][0] = CNumber.multiply(CNumber.addInv(CNumber.sign(x.entries[0][0])), norm); // TODO: Exception if m<n
				v = w.sub(x);
				
				if(!v.H().mult(v).entries[0][0].equals(CNumber.ZERO)) {
					n = CNumber.divide(two, v.H().mult(v).entries[0][0]);
				} else {
					n = CNumber.ZERO;
				}
				
				
				if(!n.equals(CNumber.ZERO)) { // Then we need a reflector.
					counth++;	// For determinant, will need to count number of reflectors used.
					H = Matrix.I(v.m).sub(v.mult(v.H()).scalMult(n));
					H = Matrix.I(A.m).setSliceCopy(j, j, H);
					
					R = H.mult(R);
					Q = Q.mult(H);
				}
			}
			
		}
		
		R.roundToZero(13); // Ensure R is triangular by rounding values near zero to zero.
		
		if(countHouseholder) { 
			// Then we also need to return the number of householder reflectors used.
			// This is stored in a 1-by-1 matrix
			QR = new Matrix[3];
			QR[2] = new Matrix(1, 1, counth);
		} else {
			QR = new Matrix[2];
		}
		
		QR[0] = Q;
		QR[1] = R;
		
		return QR;
	}
	
	
	// For testing a QR algorithm to work with complex matrices.
	protected static Matrix[] QRtemp(Matrix A, boolean countHouseholder) {
		Matrix QR[], 
				Q = Matrix.I(A.m), 
				R = A,
				H,
				x, x_hat, u_hat; // Column Vector
		
		Vector w;
		CNumber norm;
		CNumber n = CNumber.ZERO;
		int counth = 0; // Counts the number of householder Reflectors used.
		
		for(int j=0; j<A.n; j++) {
			if(!R.getSlice(j, A.m, j, j+1).equals(new Matrix(A.m-j, 1))) { // Then we need a reflector.
				counth++;
				
				x = R.getSlice(j, A.m, j, j+1);
				
				Matrix.print("x:\n", x, "\n\n");
				
				norm = x.norm();
				x_hat = x.scalDiv(norm);
				
				Matrix.print("x_hat:\n", x_hat, "\n\n");
				
				w = new Vector(A.m-j);
				w.entries[0][0] = CNumber.sign(x.entries[0][0]); // TODO: Exception if m<n
				
				Matrix.print("w:\n", w, "\n\n");
				
				u_hat = x_hat.add(w).scalDiv(x_hat.add(w).norm());
				
				Matrix.print("u_hat:\n", u_hat, "\n\n");
				
				// This factor multiplied to H should make diagonals of R real positive
				n = CNumber.multiply(CNumber.conjugate(CNumber.sign(x.entries[0][0])), CNumber.NEGATIVE_ONE);
				
				H = (Matrix.I(u_hat.m).sub(u_hat.mult(u_hat.H()).scalMult(2))).scalMult(n);
				H = Matrix.I(A.m).setSliceCopy(0+j, 0+j, H);
				
				Matrix.print("H:\n", H, "\n\n");
				
				R = H.mult(R);
				Q = Q.mult(H.H());
			}
		}
		
		R.roundToZero(13); // Ensure R is triangular by rounding values near zero to zero.
		
		if(countHouseholder) { 
			// Then we also need to return the number of householder reflectors used.
			// This is stored in a 1-by-1 matrix
			QR = new Matrix[3];
			QR[2] = new Matrix(1, 1, counth);
		} else {
			QR = new Matrix[2];
		}
		
		QR[0] = Q;
		QR[1] = R;
		
		return QR;
	}
	
	
	public static void main(String[] args) {
		double[][] a = {{0, 0, 0},
			 	 		{0, 0, -1},
			 	 		{0, 1, 0}}; // TODO: QR decomposition fails for this case
		
		int[][] a1 = { {0, 1, 0, 0},
				 	   {0, 0, 1, 0},
				 	   {0, 0, 0, 1},
				 	   {1, 0, 0, 0}}; // TODO: QR decomposition fails for this case

		CNumber[][] ac = {{new CNumber("2+i"), new CNumber("3")},
						  {new CNumber("2i"), new CNumber("-i")}};
		
		CNumber[][] ac1 = {	{new CNumber("1.2-2.5i"), 	new CNumber("i"), new CNumber("-8+4i"), new CNumber("-5"), 		new CNumber("10")},
							{new CNumber("2"), 		new CNumber("0"), new CNumber("4"), 	new CNumber("1-i"), 	new CNumber("15i")},
							{new CNumber("4"), 			new CNumber("0"), new CNumber("6"),		new CNumber("1+i"), 	new CNumber("3")},
							{new CNumber("0"), 		new CNumber("0"), new CNumber("-5-i"), 	new CNumber("9.6"), 	new CNumber("-9")},
							{new CNumber("0"), 		new CNumber("0"), new CNumber("4+9i"), 	new CNumber("4-9i"), 	new CNumber("11")}};
		
		double[][] a2 = { {   22.484806463399448,   -1.0735875467841485,     2.831257356052518,   -3.2313318895115994 },
				  		  { 		  2,			-2.9847406120860756,    	-0.0254201659129218,  	2.4690227481521076  },
						  {           30,             0,           				0,           			0.46682358571502225 },
						  {           0,             0,    						-2.41296221661233,    	3.78499875867359   } };
		
		Matrix A = new Matrix(a);
		Matrix[] QR = QR(A, true);
		
		Matrix.print("A:\n", A, "\n\n");
		Matrix.print("Q:\n", QR[0].round(10), "\n\n");
		Matrix.print("R:\n", QR[1].round(10), "\n\n");
		Matrix.print("h:\n", QR[2], "\n\n");
		Matrix.print("Q^H*Q:\n", QR[0].H().mult(QR[0]).round(10), "\n\n");
		Matrix.print("Q*R:\n", QR[0].mult(QR[1]).round(10), "\n\n");
		Matrix.print("det(A): ", A.det(), "\n\n");
		Matrix.print("----------------------------------------------------------------------\n\n");
	}
}
