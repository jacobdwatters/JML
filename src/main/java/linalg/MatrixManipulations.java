package linalg;

import linalg.complex_number.CNumber;
import linalg.util.LinAlgArrayUtils;
import linalg.util.Parser;

/**
 * This interface provides several methods useful for matrix manipulations and is
 * Implemented by the Matrix class.
 * 
 * @author Jacob Watters
 */
// TODO: Many methods in this class should probably manipulate the instance object and have void return.
 interface MatrixManipulations {
	
	
	 default Matrix reshape(String newShape) {
		int[] dimensions = Parser.parseShape(newShape);
		int 	newM = dimensions[0],
				newN = dimensions[1];
		
		return this.reshape(newM, newN);
	}
	
	
	 default Matrix reshape(int newM, int newN) {
		Matrix A = ((Matrix) this).flatten();
		Matrix reshape = new Matrix(newM, newN);
		int aj=0;

		if(newM*newN != A.m*A.n) {
			throw new IllegalArgumentException("Can not reshape matrix of shape " + A.shape + " to " + newM + "x" + newN + ".");
		}


		for(int i=0; i<reshape.m; i++) {
			for(int j=0; j<reshape.n; j++) {
				reshape.entries[i][j] = new CNumber(A.entries[0][aj]);
				aj++;
			}
		}
		
				
		return reshape;
	}
	
	
	/**
	 * Flattens an <code>MxN</code> matrix to a 1x(m*n) matrix.
	 * Each row of the Matrix is appended to the end
	 * of the first row of the Matrix in order.
	 * 
	 * @return 
	 */
	 default Matrix flatten() {
		Matrix A = (Matrix) this,
				flat;
		int count = 0;

		CNumber[][] flat_A = new CNumber[1][A.m*A.n];
		
		for(int i = 0; i < A.m; i++) {
			for(int j = 0; j < A.n; j++) {
				flat_A[0][count] = A.entries[i][j];
				count++;
			}
		}
		
		flat = new Matrix(flat_A);
		
		return flat;
	}
	
	
	/**
	 * Flattens an <code>MxN</code> matrix to a 1x(m*n) or (m*n)x1 matrix depending on axis.
	 *
	 * @param axis - axis along which to flatten
	 * @return 
	 */
	 default Matrix flatten(int axis) {
		Matrix A = (Matrix) this,
				flat;
		
		if(axis == 0) {
			flat = A.flatten();
		}
		else if(axis == 1) {	
			int count = 0;

			CNumber[][] flat_A = new CNumber[A.m*A.n][1];
			
			for(int i = 0; i < A.m; i++) {
				for(int j = 0; j < A.n; j++) {
					flat_A[count][0] = A.entries[i][j];
					count++;
				}
			}
			
			flat = new Matrix(flat_A);
		}
		else {
			throw new IllegalArgumentException("axis must be 0 or 1 but got " + axis);
		}
		
		return flat;
	}
	
	
	/**
	 * Copies values from array into matrix. The given array must have the same dimensions
	 * as the matrix object. A will replace any current values in the matrix.
	 * 
	 *  @param values - Values to copy into array
	 */
	 default void setValues(CNumber[][] values) {
		Matrix A = (Matrix) this;
		
		if(values.length != A.entries.length || values[0].length != A.entries[0].length) {
			throw new IllegalArgumentException("Expecting array with shape " + A.shape + 
					" but got array with shape " + values.length + "x" + values[0].length + ".");
		}
		
		A.entries = values;
	}
	
	
	/**
	 * Copies values from array into matrix. The given array must have the same dimensions
	 * as the matrix object. A will replace any current values in the matrix.
	 * 
	 *  @param values - Values to copy into array
	 */
	 default void setValues(double[][] values) {
		Matrix A = (Matrix) this;
		
		if(values.length != A.entries.length || values[0].length != A.entries[0].length) {
			throw new IllegalArgumentException("Expecting array with shape " + A.shape + 
					" but got array with shape " + values.length + "x" + values[0].length + ".");
		}
		
		for(int i = 0; i < A.entries.length; i++) {
			for(int j = 0; j < A.entries[0].length; j++) {
				A.entries[i][j] = new CNumber(values[i][j]);
			}
		}
	}
	
	
	/**
	 * Sets specified element in matrix to value. A will replace the current value
	 * at that index.
	 *
	 * @param value - value to insert into matrix
	 * @param row - row to insert value
	 * @param col - column to insert value
	 */
	 default void set(CNumber value, int row, int col) {
		Matrix A = (Matrix) this;
		
		A.entries[row][col] = value;
	}
	
	
	/**
	 * Sets specified element in matrix to value. A will replace the current value
	 * at that index.
	 *
	 * @param value - value to insert into matrix
	 * @param row - row index to insert value
	 * @param col - column index to insert value
	 */
	 default void set(double value, int row, int col) {
		Matrix A = (Matrix) this;
		
		A.entries[row][col] = new CNumber(value);
	}
	
	
	/**
	 * Sets specified column of this matrix to the values stored within the passed column vector.
	 * <br><br>
	 * Also see <br>
	 * {@link #setCol(CNumber[], int) setCol(CNumber[] col, int colIndex)}<br>
	 * {@link #setCol(double[], int) setCol(double[] col, int colIndex)}
	 * 
	 * @throws IllegalArgumentException if the vector is not a column vector or if the vector and matrix do not
	 * 	have the same number of rows.
	 * @param col - Column vector containing the new entries for the specified column.
	 * @param colIndex - Index of new column to set. 
	 */
	 default void setCol(Vector col, int colIndex) {
		Matrix A = (Matrix) this;
		
		if(col.vectorType() != Vector.COLUMN_VECTOR) {
			throw new IllegalArgumentException("Vector must be column vector."); // TODO: Add vectorTypeAsString();
		}
		if(col.m != A.m) {
			throw new IllegalArgumentException("Can not set column for matrix of shape " + A.shape + " with a vector of shape " + col.shape);
		}
		
		for(int i=0; i<A.m; i++) {
			A.entries[i][colIndex] = col.get(i, 0);
		}
	}
	
	
	/**
	 * Sets specified column of this matrix to the values stored within the passed array.
	 * <br><br>
	 * Also see<br>
	 * {@link #setCol(Vector, int) setCol(Vector col, int colIndex)}<br>
	 * {@link #setCol(double[], int) setCol(double[] col, int colIndex)}
	 * 
	 * @throws IllegalArgumentException if the array of values has a different length then the number of rows in the matrix.
	 * @param col - array containing the new entries for the specified column.
	 * @param colIndex - Index of new column to set. 
	 */
	 default void setCol(CNumber[] col, int colIndex) {
		Matrix A = (Matrix) this;
		
		if(col.length != A.m) {
			throw new IllegalArgumentException("Can not set column for matrix of shape " + A.shape + " with array of length " + col.length);
		}
		
		for(int i=0; i<A.m; i++) {
			A.entries[i][colIndex] = new CNumber(col[i]);
		}
	}
	
	
	/**
	 * Sets specified column of this matrix to the values stored within the passed array.
	 * Also see<br>
	 * {@link #setCol(Vector, int) setCol(Vector col, int colIndex)}<br>
	 * {@link #setCol(CNumber[], int) setCol(CNumber[] col, int colIndex)}
	 * 
	 * @throws IllegalArgumentException if the array of values has a different length then the number of rows in the matrix.
	 * @param col - array containing the new entries for the specified column.
	 * @param colIndex - Index of new column to set. 
	 */
	 default void setCol(double[] col, int colIndex) {
		Matrix A = (Matrix) this;
		
		if(col.length != A.m) {
			throw new IllegalArgumentException("Can not set column for matrix of shape " + A.shape + " with array of length " + col.length);
		}
		
		for(int i=0; i<A.m; i++) {
			A.entries[i][colIndex] = new CNumber(col[i]);
		}
	}
	
	
	/**
	 * Sets specified row of this matrix to the values stored within the passed row vector.
	 * <br><br>
	 * Also see <br>
	 * {@link #setRow(CNumber[], int) setRow(CNumber[] col, int colIndex)}<br>
	 * {@link #setRow(double[], int) setRow(double[] col, int colIndex)}
	 * 
	 * @throws IllegalArgumentException if the vector is not a row vector or if the vector and matrix do not
	 * 	have the same number of columns.
	 * @param row - row vector containing the new entries for the specified column.
	 * @param rowIndex - Index of new row to set. 
	 */
	 default void setRow(Vector row, int rowIndex) {
		Matrix A = (Matrix) this;
		
		if(row.vectorType() != Vector.ROW_VECTOR) {
			throw new IllegalArgumentException("Vector must be row vector."); // TODO: Add vectorTypeAsString();
		}
		if(row.n != A.n) {
			throw new IllegalArgumentException("Can not set column for matrix of shape " + A.shape + " with a vector of shape " + row.shape);
		}
		
		for(int i=0; i<A.n; i++) {
			A.entries[rowIndex][i] = row.get(0, i);
		}
	}
	
	
	/**
	 * Sets specified row of this matrix to the values stored within the passed array.
	 * <br><br>
	 * Also see<br>
	 * {@link #setRow(Vector, int) setRow(Vector col, int colIndex)}<br>
	 * {@link #setRow(double[], int) setRow(double[] col, int colIndex)}
	 * 
	 * @throws IllegalArgumentException if the array of values has a different length then the number of columns in the matrix.
	 * @param row - array containing the new entries for the specified column.
	 * @param rowIndex - Index of new column to set. 
	 */
	 default void setRow(CNumber[] row, int rowIndex) {
		Matrix A = (Matrix) this;
		
		if(row.length != A.n) {
			throw new IllegalArgumentException("Can not set column for matrix of shape " + A.shape + " with array of length " + row.length);
		}
		
		for(int i=0; i<A.n; i++) {
			A.entries[rowIndex][i] = new CNumber(row[i]);
		}
	}
	
	
	/**
	 * Sets specified row of this matrix to the values stored within the passed array.
	 * <br><br>
	 * Also see<br>
	 * {@link #setRow(Vector, int) setRow(Vector col, int colIndex)}<br>
	 * {@link #setRow(CNumber[], int) setRow(CNumber[] col, int colIndex)}
	 * 
	 * @throws IllegalArgumentException if the array of values has a different length then the number of columns in the matrix.
	 * @param row - array containing the new entries for the specified column.
	 * @param rowIndex - Index of new column to set. 
	 */
	 default void setRow(double[] row, int rowIndex) {
		Matrix A = (Matrix) this;
		
		if(row.length != A.n) {
			throw new IllegalArgumentException("Can not set column for matrix of shape " + A.shape + " with array of length " + row.length);
		}
		
		for(int i=0; i<A.n; i++) {
			A.entries[rowIndex][i] = new CNumber(row[i]);
		}
	}
	
	
	/**
	 * Creates a copy of this matrix and sets a specified slice of the copy to the values stored in the <code>values</code> matrix. Together
	 * the parameters <code>rowStart</code> and <code>colStart</code> define the upper left corner of the 
	 * slice to set.<br><br>
	 * 
	 * If the values matrix does not fit within this matrix, an error will be thrown.
	 * <br><br>
	 * If want to adjust this matrix instance directly see {@link #setSlice(int, int, Matrix) setSlice(Matrix values, int rowStart, int colStart)}
	 * 
	 * @return
	 */
	 default Matrix setSliceCopy(int rowStart, int colStart, Matrix values) {
		Matrix A = ((Matrix) this).copy();
		
		if(values.m + rowStart > A.m || values.n + colStart > A.n) {
			throw new IllegalArgumentException("Values do not fit in matrix. Attempting to set values of shape " + values.shape +
					" with offset of (" + rowStart + ", " + colStart + ") in a matrix of shape " + A.shape + ".");
		}
		
		for(int i=rowStart; i<values.m + rowStart; i++) {
			for(int j=colStart; j<values.n + colStart; j++) {
				A.entries[i][j] = values.entries[i-rowStart][j-colStart];
			}
		}
		
		return A;
	}
	
	
	/**
	 * Sets a specified slice of this matrix to the values stored in the <code>values</code> matrix. Together
	 * the parameters <code>rowStart</code> and <code>colStart</code> define the upper left corner of the 
	 * slice to set.<br><br>
	 * 
	 * If the values matrix does not fit within this matrix, an error will be thrown.
	 * <br><br>
	 * If you do not want to adjust this matrix instance see {@link #setSliceCopy(int, int, Matrix) setSliceCopy(Matrix values, int rowStart, int colStart)}
	 * @param rowStart - Row on original matrix to place top row of values matrix
	 * @param colStart - Column on original matrix to place left-most column of values matrix
	 * @param values - New values to set within specified slice.
	 */
	 default void setSlice(int rowStart, int colStart, Matrix values) {
		Matrix A = (Matrix) this;
		
		if(values.m + rowStart > A.m || values.n + colStart > A.n) {
			throw new IllegalArgumentException("Values do not fit in matrix. Attempting to set values of shape " + values.shape +
					" with offset of (" + rowStart + ", " + colStart + ") in a matrix of shape " + A.shape + ".");
		}
		
		for(int i=rowStart; i<values.m + rowStart; i++) {
			for(int j=colStart; j<values.n + colStart; j++) {
				A.entries[i][j] = values.entries[i-rowStart][j-colStart];
			}
		}
	}
	
	
	/**
	 * Removes specified single row from matrix.
	 * 
	 * To remove more than one row at a time
	 * see {@link #removeRows(int...) removeRows(int... rowIndices)}.
	 * 
	 * @param rowIndex - Index of row to remove.
	 * @return Matrix with the specified row removed.
	 */
	 default Matrix removeRow(int rowIndex) {
		Matrix A = (Matrix) this;
		return A.removeRows(rowIndex);
	}
	
	
	/**
	 * Removes specified rows from matrix. If k row indices are specified for 
	 * a mxn matrix, the resulting matrix will have shape (m-k)xn. 
	 * 
	 * Also see {@link #removeRow(int) removeRow(int rowIndex)}.
	 * 
	 * @param rowIndices - list of row indices to remove from matrix.
	 * @return Matrix with specified rows removed
	 */
	 default Matrix removeRows(int... rowIndices) {
		Matrix A = (Matrix) this;
		Matrix result = new Matrix(A.m-(rowIndices.length), A.n);
		
		int newi = 0;
		
		for(int i = 0; i < A.m; i++) {
			if(!LinAlgArrayUtils.contains(rowIndices, i)) {
				for(int j = 0; j < A.n; j++) {
					result.entries[newi][j] = A.entries[i][j];
				}
				newi++;
			}
		}
		
		return result;
	}
	
	
	/**
	 * Removes specified single column from matrix.
	 * 
	 * To remove more than one column at a time
	 * see {@link #removeCols(int...) removeCols(int... colIndices)}.
	 * 
	 * @param colIndex - Index of column to remove.
	 * @return Matrix with the specified column removed.
	 */
	 default Matrix removeCol(int colIndex) {
		Matrix A = (Matrix) this;
		return A.removeCols(colIndex);
	}
	
	
	/**
	 * Removes specified columns from matrix. If k column indices are specified for 
	 * a mxn matrix, the resulting matrix will have shape m-by-(n-k).
	 *
	 * Also see {@link #removeCol(int) removeCol(int colIndex)}
	 * 
	 * @param colIndices - list of column indices to remove from matrix.
	 * @return Matrix with specified columns removed
	 */
	 default Matrix removeCols(int... colIndices) {
		Matrix A = (Matrix) this;
		Matrix result = new Matrix(A.m, A.n-(colIndices.length));
		
		int newj = 0;
		
		for(int j = 0; j < A.n; j++) {
			if(!LinAlgArrayUtils.contains(colIndices, j)) {
				for(int i = 0; i < A.m; i++) {
					result.entries[i][newj] = A.entries[i][j];
				}
				newj++;
			}
		}
		
		return result;
	}
	
	
	/**
	 * Generates lower triangle of Matrix. That is, all values above the middle
	 * diagonal will be zeroed.
	 * 
	 * Also see <code>{@link #tril(int) tril(int k)}</code>
	 * 
	 * @return Lower triangle of Matrix.
	 */
	 default Matrix tril() {
		return this.tril(0);
	}
	
	
	/**
	 * Generates lower triangle of Matrix. That is, all values above the kth
	 * diagonal will be zeroed.
	 * 
	 * Also see <code>{@link #tril(int) tril()}</code>
	 * 
	 * @param k - diagonal above which to zero. k=0 whould be middile diagonal, k=-1 would be 
	 * 	the diagonal to the left of the middle diagonal, and k=1 would be the diagonal to the right
	 * 	of the middle diagonal.
	 * @return Lower triangle of Matrix.
	 */
	 default Matrix tril(int k) {
		Matrix A = (Matrix) this;
		Matrix lower = Matrix.zeros(A.m, A.n);
		
		for(int i = 0; i < lower.m; i++) {
			for(int j = 0; j<=(i+k) && j < lower.n; j++) {
				lower.entries[i][j] = A.entries[i][j];
			}
		}
		
		return lower;
	}
	
	
	/**
	 * Generates upper triangle of Matrix. That is, all values above the middle
	 * diagonal will be zeroed.
	 * 
	 * Also see <code>{@link #triu(int) triu(int k)}</code>
	 * 
	 * @return upper triangle of Matrix.
	 */
	 default Matrix triu() {
		return this.triu(0);
	}
	
	
	/**
	 * Generates upper triangle of Matrix. That is, all values below the kth
	 * diagonal will be zeroed.
	 * 
	 * Also see <code>{@link #triu(int) triu()}</code>
	 * 
	 * @param k - diagonal above which to zero. k=0 would be middle diagonal, k=-1 would be 
	 * 	the diagonal to the left of the middle diagonal, and k=1 would be the diagonal to the right
	 * 	of the middle diagonal.
	 * @return upper triangle of Matrix.
	 */
	 default Matrix triu(int k) {
		Matrix A = (Matrix) this;
		Matrix upper = Matrix.zeros(A.m, A.n);
		int jstart = 0;
		
		
		for(int i = 0; i < upper.m; i++) {
			if(i+k <= 0) {jstart = 0;}
			else {jstart = i+k;}
			
			for(int j = jstart; j-k>=(i) && j < upper.n; j++) {
				upper.entries[i][j] = A.entries[i][j];
			}
		}
		
		return upper;
	}
	
	
	/**
	 * Extracts diagonal elements from matrix.
	 * 
	 * @return Returns an equivalently sized matrix containing only the diagonal elements of this matrix.
	 */
	 default Matrix diag() {
		return this.triu(0).tril(0);
	}
	
	
	/**
	 * Sets elements from list as diagonal elements of a zero matrix.
	 * 
	 * @return Returns an equivalently sized matrix containing only the diagonal elements of this matrix.
	 */
	 static Matrix toDiag(CNumber... entries) {
		Matrix A = new Matrix(entries.length);
		
		for(int i=0; i<entries.length; i++) {
			A.entries[i][i] = entries[i];
		}
		
		return A;
	}
	
	
	/**
	 * Extracts diagonal elements form matrix and stores in vector.
	 * 
	 * @return Column vector containing diagonal elements of this matrix.
	 */
	 default Vector diagAsVector() {
		Matrix A = (Matrix) this;
		Vector diag = new Vector(Math.min(A.m, A.n));
		
		for(int i=0; i<diag.m; i++) {
			diag.entries[i][0] = A.entries[i][i];
		}
		
		return diag;
	}
	
	
	/**
	 * Converts a matrix similar to this matrix that is in upper Hessenburg form. A matrix is in upper Hessenburg form if all entries below the first subdiagonal are
	 * zero. Two n-by-n matrices A and B are similar if there exists an invertible n-by-n matrix P, such that B=P<sup>-1</sup>AP. Similar matrices share many 
	 * properties including the same eigenvalues.
	 * 
	 * @return A matrix similar to this matrix which is in upper Hessenburg form.
	 */
	 default Matrix hessu() {
		Matrix H = ((Matrix) this).copy();
		Matrix x, v = new Matrix(H.m, H.m);
		CNumber two = new CNumber(2), eps = new CNumber(Float.MIN_VALUE);
		
		if(!H.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got shape " + H.shape);
		}
		
		
		for(int k=0; k<H.m-2; k++) {	
			if(!H.getSlice(k+1, H.m, k, k+1).equals(new Matrix(H.m-(k+1), 1))) { // Then we need a Householder reflector. Otherwise, the column is already in the correct form so we don't need to apply a reflector. 
				x = H.getSlice(k+1, H.m, k, k+1);
				
				v.setSlice(0, k, 	
						Matrix.I(H.m-(k+1), 1).scalMult(
						CNumber.multiply(
							CNumber.addInv(CNumber.sign(CNumber.add(x.entries[0][0], eps))), 
							x.norm()
						)).sub(x)
				);
				
				v.setSlice(0, k, v.getSlice(0, H.m-(k+1), k, k+1).scalDiv(v.getSlice(0, H.m-(k+1), k, k+1).norm()));
				
				H.setSlice(k+1, k, 
						H.getSlice(k+1, H.m, k, H.m).sub(v.getSlice(0, H.m-(k+1), k, k+1).mult(
						v.getSlice(0, H.m-(k+1), k, k+1).H().mult(H.getSlice(k+1, H.m, k, H.m))).scalMult(two))
				);
				
				H.setSlice(0, k+1, 
						H.getSlice(0, H.m, k+1, H.m).sub(H.getSlice(0, H.m, k+1, H.m).mult(
						v.getSlice(0, H.m-(k+1), k, k+1)).mult(v.getSlice(0, H.m-(k+1), k, k+1).H()).scalMult(two))
				);
			}
		}
		
		H.roundToZero(10);
		
		return H;
	}
	
	
	/**
	 * Converts a matrix similar to this matrix that is in upper Hessenburg form. A matrix is in upper Hessenburg form if all entries below the first subdiagonal are
	 * zero. Two n-by-n matrices A and B are similar if there exists an invertible n-by-n matrix P, such that B=P<sup>-1</sup>AP. Similar matrices share many 
	 * properties including the same eigenvalues.
	 * 
	 * @return A matrix similar to this matrix which is in upper Hessenburg form.
	 */
	 default Matrix[] hessuV() {
		Matrix H = ((Matrix) this).copy();
		Matrix x, v = new Matrix(H.m, H.m);
		CNumber two = new CNumber(2), eps = new CNumber(Float.MIN_VALUE);
		Matrix[] result = new Matrix[2];
		
		if(!H.isSquare()) {
			throw new IllegalArgumentException("Matrix must be square but got shape " + H.shape);
		}
		
		
		for(int k=0; k<H.m-2; k++) {	
			x = H.getSlice(k+1, H.m, k, k+1);
			
			v.setSlice(0, k, 	
					Matrix.I(H.m-(k+1), 1).scalMult(
					CNumber.multiply(
						CNumber.addInv(CNumber.sign(CNumber.add(x.entries[0][0], eps))), 
						x.norm()
					)).sub(x)
			);
			
			v.setSlice(0, k, v.getSlice(0, H.m-(k+1), k, k+1).scalDiv(v.getSlice(0, H.m-(k+1), k, k+1).norm()));
			
			H.setSlice(k+1, k, 
					H.getSlice(k+1, H.m, k, H.m).sub(v.getSlice(0, H.m-(k+1), k, k+1).mult(
					v.getSlice(0, H.m-(k+1), k, k+1).H().mult(H.getSlice(k+1, H.m, k, H.m))).scalMult(two))
			);
			
			H.setSlice(0, k+1, 
					H.getSlice(0, H.m, k+1, H.m).sub(H.getSlice(0, H.m, k+1, H.m).mult(
					v.getSlice(0, H.m-(k+1), k, k+1)).mult(v.getSlice(0, H.m-(k+1), k, k+1).H()).scalMult(two))
			);
		}
		
		H.roundToZero(10);
		
		result[0] = H;
		result[1] = v;
		
		return result;
	}
	
	
	/**
	 * Swaps two rows in a matrix.
	 * 
	 * @param rowIndex1 - Index of first row to be swapped.
	 * @param rowIndex2 - Index of second row to be swapped.
	 * @return Matrix with specified rows swapped.
	 */
	 default Matrix swapRows(int rowIndex1, int rowIndex2) {
		Matrix A = (Matrix) this;
		Matrix result = A.copy();
		
		CNumber[] row1 = A.entries[rowIndex1];
		CNumber[] row2 = A.entries[rowIndex2];
		
		result.entries[rowIndex1] = row2;
		result.entries[rowIndex2] = row1;
		
		return result;
	}
	
	
	/**
	 * Swaps two columns in a matrix.
	 * 
	 * @param colIndex1 - Index of first column to be swapped.
	 * @param colIndex2 - Index of second column to be swapped.
	 * @return Matrix with specified columns swapped.
	 */
	 default Matrix swapCols(int colIndex1, int colIndex2) {
		Matrix A = (Matrix) this;
		Matrix result = A.copy();
		
		CNumber[] col1 = A.getCol(colIndex1);
		CNumber[] col2 = A.getCol(colIndex2);
		
		for(int i = 0; i<A.m; i++) {
			result.entries[i][colIndex1] = col2[i];
			result.entries[i][colIndex2] = col1[i];
		}
		
		return result;
	}
	
	
	/**
	 * Multiplies a specified row by a constant value.
	 * 
	 * @param factor - number to multiply row by.
	 * @return Matrix with specified row multiplied by the factor.
	 */
	 default Matrix multRow(int rowIndex, CNumber factor) {
		Matrix A = (Matrix) this;
		Matrix result = new Matrix(A.m, A.n);
		
		for(int j=0; j<A.n; j++) {
			result.entries[rowIndex][j] = CNumber.multiply(A.entries[rowIndex][j], factor); 
		}
		
		return result;
	}

	
	/**
	 * Multiplies a specified row by a constant value.
	 * 
	 * @param factor - number to multiply row by.
	 * @return Matrix with specified row multiplied by the factor
	 */
	 default Matrix multRow(int rowIndex, double factor) {
		return multRow(rowIndex, new CNumber(factor));
	}
	
	
	/**
	 * Divides a specified row by a constant value.
	 * 
	 * @param divisor - number to divide row by.
	 * @return Matrix with specified row divided by the divisor.
	 */
	 default Matrix divRow(int rowIndex, CNumber divisor) {
		return multRow(rowIndex, CNumber.inv(divisor));
	}
	
	
	/**
	 * Divides a specified row by a constant value.
	 * 
	 * @param divisor - number to divide row by.
	 * @return Matrix with specified row divided by the divisor.
	 */
	 default Matrix divRow(int rowIndex, double divisor) {
		return multRow(rowIndex, 1 / divisor);
	}
	
	
	/**
	 * If possible, converts matrix to a Vector object.<br><br>
	 * A matrix will be converted to a row vector if it contains only a single row. <br>
	 * A matrix will be converted to a column vector if it contains only a single column. <br>
	 * If a matrix contains only one row and one column or is empty, then it will be converted to a column vector by default.<br>
	 * 
	 * @return Vector equivalent of matrix.
	 */
	 default Vector toVector() {
		Matrix A = (Matrix) this;
		Vector result = null;
		
		if(!A.isVector() && !A.isEmpty()) {
			throw new IllegalArgumentException("Can not convert matrix of shape " + A.shape + " to a vector.");
		} else {
			int vectorType = A.vectorType();
			
			
			if(A.isEmpty()) {
				result = new Vector();
			}
			else { // 1-by-1 matrix or empty matrix will default to a column vector.
				if(vectorType == 3) {
					vectorType = Vector.COLUMN_VECTOR;
				}
				
				result = new Vector(LinAlgArrayUtils.flatten(A.entries), vectorType);
			}
		}
		
		return result;
	}
	
	
	/**
	 * Rounds all entries of a matrix. If entry is complex, 
	 * both the real and imaginary part will be rounded.
	 *
	 * @param decimals - Number of decimals to round number to.
	 * @return Returns copy of Matrix <code>A</code> with entries rounded to 
	 * 		Specified number of decimal places.  
	 */
	 default Matrix round(int decimals) {
		Matrix rounded = new Matrix((Matrix) this);
		
		for(int i=0; i<rounded.m; i++) {
			for(int j=0; j<rounded.n; j++) {
				rounded.entries[i][j] = CNumber.round(rounded.entries[i][j], decimals);
			}
		}
		
		return rounded;
	}
	
	
	/**
	 * Rounds values in this matrix within a specified number of decimals of zero to zero.
	 * 
	 * @param decimals - Number of decimals 
	 */
	 default void roundToZero(int decimals) {
		double tolerance = 0.5*Math.pow(10, -decimals);
		Matrix A = (Matrix) this;
		
		for(int i=0; i<A.m; i++) {
			for(int j=0; j<A.n; j++) {
				if(Math.abs(A.entries[i][j].mag()) < tolerance) {
					A.entries[i][j] = CNumber.ZERO;
				}
			}
		}
	}
	
	
	 static void main(String[] args) {
		
		double[] val = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,1, 12, 12};
		double[][] 	a =  {{2, 5, 8, 7},
				 		  {5, 2, 2, 8},
						  {7, 5, 6, 6},
						  {5, 4, 4, 8}};
		                       
		CNumber[][] ac = {	{new CNumber("2+2i"),	new CNumber("5"), new CNumber("8"), new CNumber("7")},
							{new CNumber("5"), 		new CNumber("i"), new CNumber("2"), new CNumber("8")},
							{new CNumber("7"),		new CNumber("5"), new CNumber("6"), new CNumber("6")},
							{new CNumber("5"), 		new CNumber("4"), new CNumber("4"), new CNumber("8")} };
		
		
		Matrix A = new Matrix(a);
		Vector x = new Vector(val);
		
		Matrix.println("V:\n", Matrix.van(x, 10), "\n\n");
	}
}









