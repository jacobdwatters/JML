package linalg;

import linalg.complex_number.CNumber;
import linalg.util.LinAlgArrayUtils;


/**
 * This class supports the creation, manipulations, and operations of Vectors.
 */
public class Vector extends Matrix {
	public static final int COLUMN_VECTOR = 0,
								ROW_VECTOR = 1;

	private static final String INVALID_TYPE_ERR = "Type must be either " + COLUMN_VECTOR + " or "
			+ ROW_VECTOR + " but got ";
	protected int type;
	protected int length;
	
	/**
	 * Creates an empty column vector.
	 */
	public Vector() {
		this.entries = new Matrix().getValues();
		this.m = 0;
		this.n = 0;
		this.shape = m + "x" + n;
		this.type = COLUMN_VECTOR;
		this.length = 0;
	}
	
	/**
	 * Creates a column vector of specified size filled with zeros.
	 * @param size - size of the column vector
	 */
	public Vector(int size) {
		this.entries = new Matrix(size, 1).getValues();
		this.m = size;
		this.n = 1;
		this.shape = m + "x" + n;
		this.type = COLUMN_VECTOR;
		this.length = size;
	}
	
	
	/**
	 * Creates a row/column vector of specified size filled with zeros.
	 * 
	 * @param size - Size of the vector.
	 * @param type - Specifies the type of vector. Pass a 0 for a column vector
	 * 		or a 1 for a row vector. 
	 */
	public Vector(int size, int type) {
		if(type == ROW_VECTOR) {
			this.entries = new Matrix(1, size).getValues();
			this.m = 1;
			this.n = size;
		} else if(type == COLUMN_VECTOR) {
			this.entries = new Matrix(size, 1).getValues();
			this.m = size;
			this.n = 1;
		} else {
			throw new IllegalArgumentException(INVALID_TYPE_ERR + type + ".");
		}
		
		this.shape = m + "x" + n;
		this.type = type;
		this.length = size;
	}
	
	
	/**
	 * Creates a column vector from the entries array.
	 * 
	 * @param entries - entries of the column vector.
	 */
	public Vector(int[] entries) {
		this.entries = LinAlgArrayUtils.extend2D(entries, 1);
		this.m = entries.length;
		this.n = 1;
		this.shape = m + "x" + n;
		this.type = COLUMN_VECTOR;
		this.length = entries.length;
	}
	
	
	/**
	 * Creates a row/column vector depending on the value passed to <code>type</code>.
	 * 
	 * @param entries - entries of the vector.
	 * @param type - Specifies the type of vector. Pass a 0 for a column vector
	 * 		or a 1 for a row vector. 
	 */
	public Vector(int[] entries, int type) {
		if(type == ROW_VECTOR) {
			this.entries = LinAlgArrayUtils.extend2D(entries, 0);
			this.n = entries.length;
			this.m = 1;
			this.shape = m + "x" + n;
			this.type = ROW_VECTOR;
		} else if(type == COLUMN_VECTOR) {
			this.entries = LinAlgArrayUtils.extend2D(entries, 1);
			this.m = entries.length;
			this.n = 1;
			this.shape = m + "x" + n;
			this.type = COLUMN_VECTOR;
		} else {
			throw new IllegalArgumentException(INVALID_TYPE_ERR + type + ".");
		}
		
		this.length = entries.length;
	}
	
	
	/**
	 * Creates a column vector from the entries array.
	 * 
	 * @param entries - entries of the column vector.
	 */
	public Vector(double[] entries) {
		this.entries = LinAlgArrayUtils.extend2D(entries, 1);
		this.m = entries.length;
		this.n = 1;
		this.shape = m + "x" + n;
		this.type = COLUMN_VECTOR;
		this.length = entries.length;
	}
	
	
	/**
	 * Creates a row/column vector depending on the value passed to <code>type</code>.
	 * 
	 * @param entries - entries of the vector.
	 * @param type - Specifies the type of vector. Pass a 0 for a column vector
	 * 		or a 1 for a row vector. 
	 */
	public Vector(double[] entries, int type) {
		if(type == ROW_VECTOR) {
			this.entries = LinAlgArrayUtils.extend2D(entries, 0);
			this.n = entries.length;
			this.m = 1;
			this.shape = m + "x" + n;
			this.type = ROW_VECTOR;
		} else if(type == COLUMN_VECTOR) {
			this.entries = LinAlgArrayUtils.extend2D(entries, 1);
			this.m = entries.length;
			this.n = 1;
			this.shape = m + "x" + n;
			this.type = COLUMN_VECTOR;
		} else {
			throw new IllegalArgumentException(INVALID_TYPE_ERR + type + ".");
		}
		
		this.length = entries.length;
	}
	
	
	/**
	 * Creates a column vector from the entries array.
	 * 
	 * @param entries - entries of the column vector.
	 */
	public Vector(CNumber[] entries) {
		this.entries = LinAlgArrayUtils.extend2D(entries, 1);
		this.m = entries.length;
		this.n = 1;
		this.shape = m + "x" + n;
		this.type = COLUMN_VECTOR;
		this.length = entries.length;
	}
	
	
	/**
	 * Creates a row/column vector depending on the value passed to <code>type</code>.
	 * 
	 * @param entries - entries of the vector.
	 * @param type - Specifies the type of vector. Pass a 0 for a column vector
	 * 		or a 1 for a row vector. 
	 */
	public Vector(CNumber[] entries, int type) {
		if(type == ROW_VECTOR) {
			this.entries = LinAlgArrayUtils.extend2D(entries, 0);
			this.n = entries.length;
			this.m = 1;
			this.shape = m + "x" + n;
			this.type = ROW_VECTOR;
			
		} else if(type == COLUMN_VECTOR) {
			this.entries = LinAlgArrayUtils.extend2D(entries, 1);
			this.m = entries.length;
			this.n = 1;
			this.shape = m + "x" + n;
			this.type = COLUMN_VECTOR;
			
		} else {
			throw new IllegalArgumentException(INVALID_TYPE_ERR + type + ".");
		}
		
		this.length = entries.length;
	}

	
	public Vector(Vector a) {
		this.entries = a.entries;
		this.m = a.m;
		this.n = a.n;
		this.shape = m + "x" + n;
		this.type = a.vectorType();
		this.length = entries.length;
	}
	

	/**
	 * Converts a vector to a matrix.
	 * 
	 * @param v - vector to convert to matrix
	 * @return Matrix of vector. If the vector is a row vector the matrix will have one row and the
	 * 		same number of columns as entries in the vector. If the vector is a column vector the matrix will have one column and
	 * 		the same number of rows as entries in the vector.
	 */
	public static Matrix toMatrix(Vector v) {
		return new Matrix(v.entries);
	}
	
	
	/**
	 * Converts Vector to a row vector. 
	 * 
	 * @return If the vector is already a row vector, then the same vector is returned.
	 * 	 	If the vector is a column vecctor, then a new row vector with identical entires
	 * 		to the column vector is returned.
	 */
	public Vector toRowVector() {
		if(this.type == ROW_VECTOR) {
			return this;
		} else {
			return this.T().getRowAsVector(0);
		}
	}
	
	
	/**
	 * Converts Vector to a column vector. 
	 * 
	 * @return If the vector is already a column vector, then the same vector is returned.
	 * 	 	If the vector is a row vecctor, then a new column vector with identical entires
	 * 		to the row vector is returned.
	 */
	public Vector toColVector() {
		if(this.type == COLUMN_VECTOR) {
			return this;
		} else {
			return this.T().getColAsVector(0);
		}
	}
	
	
	/**
	 * Get the type of this vector
	 * 
	 * @return Returns 0 for a column vector, 1 for a row vector.
	 */
	public int vectorType() {
		return type;
	}
	
	
	/**
	 * Computs inner prodcut of two vectors. Note, vectors do not have to
	 * be the same type. That is, any mixture of row and column vectors is fine
	 * as long as they have the same number of entries. 
	 * 
	 * @param b - Vector to compute inner product with.
	 * @return Inner product of this vector with b.
	 */
	public CNumber innerProduct(Vector b) {
		if(this.length != b.length) {
			throw new IllegalArgumentException("Vectors must have same size but got " + this.length + " and "
					+ b.length + ".");
		}
		
		CNumber result = CNumber.ZERO;
		CNumber[][] bNew;
		
		if(this.type != b.type) {
			bNew = b.T().entries;
		} else {
			bNew = b.entries;
		}
		
		if(this.type == ROW_VECTOR) {
			for(int i = 0; i<this.length; i++) {
				result = CNumber.add(result, 
						CNumber.multiply(this.entries[0][i], 
								CNumber.conjugate(bNew[0][i])));
			}
		} else {
			for(int i = 0; i<this.length; i++) {
				result = CNumber.add(result, 
						CNumber.multiply(this.entries[i][0], 
								CNumber.conjugate(bNew[i][0])));
			}
		}
		
		return result;
	}
	
	
	/**
	 * Computes outer product of two vectors. The result of vector outer 
	 * products is a matrix.
	 * <br><br>
	 * Vectors must be of oposite types. That is, exactly 
	 * one of the two vectors must be a row vector and exactly 
	 * one of the two vectors must be a column vector.
	 * <br><br>
	 * In this method, vectors are treated as matrices with
	 * 1 row or 1 column depending on the vector type. Then,
	 * {@link #mult(Matrix) Matrix.mult(Matrix B)} is used.
	 * 
	 * @param b - Vector to compute outer product with.
	 * @return Outer product of this vector with b.
	 */
	public Matrix outerProduct(Vector b) {
		return this.mult(b);
	}
	
	
	/**
	 * Computes 2-norm (Euclidian norm) of vector.<br><br>
	 * Also see <br>
	 *  - {@link #norm(double) norm(double p)}<br>
	 *  - {@link #infNorm() infNorm()}
	 *  
	 * @return 2-norm of this vector. This will be a real value.
	 */
	public CNumber norm() {
		return norm(2);
	}
	
	
	/**
	 * Computes the p-norm of a vector. If p=1, this is called the Taxicab norm or Manhattan norm.
	 * <br><br>
	 * If p=2, this is equivalent to {@link #norm() norm()}<br> which is the euclidian norm.
	 * If p is <code>Double.POSITIVE_INFINITY</code> this is equivalent to 
	 * 
	 * @param p - norm value. Must satisfy p >= 1 and can be <code>Double.POSITIVE_INFINITY</code>. 
	 * @return
	 */
	public CNumber norm(double p) {
		if(p < 1 || Double.isNaN(p) || p == Double.NEGATIVE_INFINITY) {
			throw new IllegalArgumentException("Value must greater than or equal to 1 but got " + p);
		}
		else if(p == Double.POSITIVE_INFINITY) { // Then we compute the infinity norm
			return this.infNorm();
		}
		else {
			CNumber[] values = LinAlgArrayUtils.flatten(this.entries);
			double result = 0;
			
			for(CNumber val : values) { // Compute the standard p-norm
				result += Math.pow(val.mag(), p);
			}
			
			return new CNumber(Math.pow(result, 1/p));
		}
	}
	
	
	/**
	 * Computes the infinity or maximum norm. That is, the value with the maximum magnitude.
	 * If the vector is real this is equivalent to the maximum absolute value.
	 *  
	 * <br><br>
	 * Also see<br>
	 * 		- {@link #norm(double) norm(double p)}<br>
	 * 		- {@link #norm() norm(double)}
	 * 
	 * @return The infinity or maximum norm of this vector.
	 */
	public CNumber infNorm() {
		CNumber[] values = LinAlgArrayUtils.flatten(this.entries);
		
		return new CNumber(CNumber.max(values).mag());
	}
	
	
	/**
	 * 
	 * @return
	 */
	public CNumber[] getEntries() {
		if(this.type == ROW_VECTOR) {
			return this.entries[0];
		} else {
			return this.T().entries[0];
		}
	}
	
	
	/**
	 * Adds two vectors element-wise and stores in a new vector. Vectors must be of the same type.
	 * 
	 * @param b - Vector to add to this vector
	 * @return Result of vector addition.
	 */
	public Vector add(Vector b) {
		return new Vector(LinAlgArrayUtils.flatten(super.add(b).entries), b.type);
	}
	
	
	/**
	 * subtracts two vectors element-wise and stores in a new vector. Vectors must be of the same type.
	 * 
	 * @param b - Vector to subtract 
	 * @return Result of vector subtraction.
	 */
	public Vector sub(Vector b) {
		return new Vector(LinAlgArrayUtils.flatten(super.sub(b).entries), b.type);
	}
	
	
	public Vector scalDiv(CNumber value) {
		return new Vector(LinAlgArrayUtils.flatten(super.scalDiv(value).entries), this.type);
	}
	
	public Vector scalDiv(double value) {
		return scalDiv(new CNumber(value));
	}


	/**
	 * Converts a vector to a like matrix object
	 * @return
	 */
	public Matrix toMatrix() {
		Matrix m = new Matrix(this.m, this.n);

		if(this.type == ROW_VECTOR) {
			for(int j=0; j<this.n; j++) {
				m.entries[0][j] = this.entries[0][j];
			}

		} else if(this.type == COLUMN_VECTOR) {
			for(int i=0; i<this.m; i++) {
				m.entries[i][0] = this.entries[i][0];
			}

		} else {
			throw new IllegalArgumentException("Vector is of unknown type.");
		}

		return m;
	}
	
	/**
	 * Formats vector as a string.
	 * 
	 * @return This vector formated as a string
	 */
	public String toString() {
		String vectorString = super.toString();
		return " " + vectorString.substring(1, vectorString.length()-1);
	}
}
