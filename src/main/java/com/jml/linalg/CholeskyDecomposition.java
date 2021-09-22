package com.jml.linalg;

import com.jml.linalg.complex_number.CNumber;

class CholeskyDecomposition {
	// TODO: Doesn't seem to fully work for complex matrices
	protected static Matrix[] choleskyDecomp(Matrix A) {
		Matrix[] ch = {new Matrix(A.m), new Matrix(A.m)};
		
		for (int i = 0; i < A.n; i++) {
		    for (int j = 0; j <= i; j++) {
		        CNumber sum = CNumber.ZERO;
		        
		        for (int k = 0; k < j; k++)
		            sum = 	CNumber.add(
		            			sum,
		            			CNumber.multiply(ch[0].get(i, k), ch[0].get(j, k))
		            		);
		        if (i == j) {
		        	ch[0].set(CNumber.sqrt(CNumber.subtract(A.get(i, i), sum)), i, j);
		        } 
		        else {
		        	if(ch[0].entries[j][j].mag() > 0) {
		        		ch[0].set(
			        			CNumber.divide(
			        					CNumber.subtract(A.get(i, j), sum),
			        					ch[0].get(j, j)
			        			),
			        		i, j
			        	);
		        	}
		        }
		    }
		}
		
		ch[1] = ch[0].H();
		
		return ch;
	}
}
