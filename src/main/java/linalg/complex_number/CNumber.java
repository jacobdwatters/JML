package linalg.complex_number;

import linalg.Matrix;
import linalg.util.Parser;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;


/**
 * This class allows for the creation of complex and real numbers and provides
 * methods for computation with those numbers.
 * 
 * @author Jacob Watters
 */
// TODO: Refactor so that binary operations are not static and preformed like A.add(B) rather then CNumber.add(A, B)
public class CNumber {
	private final static java.util.Random random = new java.util.Random();
	
	/**
	 * re is real portion of complex number. im is imaginary portion of complex number.
	 */
	public double re, im;

	/**
	 * The number 1.
	 */
	public final static CNumber ONE = new CNumber(1);
	
	/**
	 * The number -1.
	 */
	public final static CNumber NEGATIVE_ONE = new CNumber(-1);
	
	/**
	 * The number 0.
	 */
	public final static CNumber ZERO = new CNumber(0); // the additive identity
	
	/**
	 * The imaginary unit Defined as i^2=-1 => i=(-1)^(1/2).
	 */
	public final static CNumber IMAGINARY_UNIT = new CNumber("i");
	
	/**
	 * The double value closer than any other to the mathmatical constant pi = 3.14159...
	 */
	public final static CNumber PI = new CNumber(Math.PI);
	
	/**
	 * The double value closer than any other to Euler's Constant e = 2.71828...
	 */
	public final static CNumber E = new CNumber(Math.exp(1));
	
	/**
	 * A real number holding the largest positive finite value of type double, (2-2^(-52))^21023
	 */
	public final static CNumber MAX_VALUE = new CNumber(Double.MAX_VALUE);
	
	/**
	 * A real number holding the smallest possible double value,  2^(-1074)
	 */
	public final static CNumber MIN_VALUE = new CNumber(Double.MIN_VALUE);
	
	// TODO: Do Javadoc for constructors
	
	public CNumber() {
		re = im = 0;
	}

	
	public CNumber(double real) {
		this.re = real;
		im = 0;
	}

	
	public CNumber(double real, double imaginary) {
		this.re = real;
		this.im = imaginary;
	}

	
	public CNumber(String num) {
		double[] content = Parser.parseNumber(num);

		re = content[0];
		im = content[1];
	}

	
	public CNumber(CNumber n) {
		this.re = n.re;
		this.im = n.im;
	}


	/**
	 * Generates a random real or complex number a, or a+bi where a and b are
	 * normaly distributed psedorandom numbers with a mean of zero and standard deviation of one.
	 * 
	 * @param complex - selects if a real or complex number should be selected..
	 * @return If <code>complex</code> false, a random real number is returned. If <code>complex</code> true, a random real and imaginary component are chosen
	 * and a complex number is returned.
	 */
	public static CNumber randn(boolean complex) {
		double real = random.nextGaussian();
		double imaginary;
		
		if(complex) {
			imaginary = random.nextGaussian();
			return new CNumber(real, imaginary);
		} else {
			return new CNumber(real);
		}
	}
	
	
	/**
	 * Generates a random real a where a is a
	 * normaly distributed psedorandom number with a mean of zero and standard deviation of one.
	 * 
	 * @return A random real number from a standard normal distribution.
	 */
	public static CNumber randn() {
		return randn(false);
	}
	
	// TODO: Should generate some random complex numbers where the real and complex quantities are random.
	/**
	 * Generates a random real number.
	 * 
	 * @return random real number
	 */
	public static CNumber random() {
		return new CNumber(Math.random());
	}

	
	/**
	 * Generates a random complex number with given magnitude.
	 * Note: the magnitude must be a non-negative real number.
	 * 
	 * @param mag - magnitude
	 * @return random complex number with specified magnitude.
	 */
	public static CNumber random(double mag) {
		
		if(mag < 0) {
			throw new IllegalArgumentException("Magnitude must be non-negative.");
		}
		
		// By Pythagorean theorem, this will result in a complex number with specified magnitude 
		double real = Math.random() * (mag);
		double imaginary = Math.sqrt(Math.pow(mag, 2) - Math.pow(real, 2));

		CNumber[] result_list = {new CNumber(real, imaginary),  // 1st quadrent result
								new CNumber(-real, imaginary),  // 2nd quadrent result
								new CNumber(-real, -imaginary),  // 3rd quadrent result
								new CNumber(real, -imaginary)}; // 4th quadrent result
		
		return result_list[random.nextInt(4)]; // Choose random 
	}

	
	/**
	 * Generates a random number between min and max. 
	 * 
	 * If magnitude_flag is passed a true, then a random complex number with magnitude
	 * between min and max (where min and max are non-negative values) is generated.
	 * 
	 * If magnitude_flag is passed a false, then a random real value between min and max is 
	 * generated. 
	 * 
	 * If no magnitude_flag is passed, then it is treated as false. 
	 * 
	 * @param min - minimum value for random number
	 * @param max - maximum value for random number
	 * @param magnitude_flag - optional flag to indicate if the Number should be real or complex.
	 * @return random real or complex number between min and max. 
	 */
	public static CNumber random(double min, double max, boolean... magnitude_flag) {
		if(magnitude_flag.length > 1) {
			throw new IllegalArgumentException("Can have at most one optional flag but got " + magnitude_flag.length);
		}
		
		if(min > max) {
			throw new IllegalArgumentException("min must be less than or equal to max but recieved "
					+ "min: " + min + " and max: " + max);
		}
		
		if (magnitude_flag.length > 0 && magnitude_flag[0] == true) {
			if(min < 0 || max < 0) {
				throw new IllegalArgumentException("For complex numbers, min and max must be non-negative values but recieved "
						+ "min: " + min + " and max: " + max);
			}
			
			double mag = Math.random() * (max - min) + min;
			return random(mag);
		} else {
			return new CNumber(Math.random() * (max - min) + min);
		}
	}

	
	/**
	 * Computes and returns complex conjugate of a complex number. For example the
	 * complex conjugate of <code>a+bi</code> is <code>a-bi</code>.
	 * 
	 * @param a - complex number
	 * @return complex conjugate of <code>a</code>
	 */
	public static CNumber conjugate(CNumber a) {
		return new CNumber(a.re, -a.im);
	}

	
	/**
	 * Performs complex addition of two numbers.
	 * 
	 * @param a - complex number
	 * @param b - complex number
	 * @return sum of a and b. e.g. a+b
	 */
	public static CNumber add(CNumber a, CNumber b) {
		return new CNumber(a.re + b.re, a.im + b.im);
	}

	
	/**
	 * Performs complex addition of two or more numbers.
	 * 
	 * @param a - complex number
	 * @param b - complex number
	 * @param values - Optional list of additional values to sum
	 * @return Sum of complex numbers. 
	 */
	public static CNumber add(CNumber a, CNumber b, CNumber... values) {
		
		double realSum = a.re + b.re;
		double imaginarySum = a.im + b.im;
		
		for(int i=0; i < values.length; i++) {
			realSum += values[i].re;
			imaginarySum += values[i].im;
		}
		
		return new CNumber(realSum, imaginarySum);
	}
	
	
	/**
	 * Performs complex subtraction of two numbers.
	 * 
	 * @param a - complex number
	 * @param b - complex number
	 * @return difference of a and b. e.g. a-b
	 */
	public static CNumber subtract(CNumber a, CNumber b) {
		return new CNumber(a.re - b.re, a.im - b.im);
	}

	
	/**
	 * Performs complex multiplication of two numbers.
	 * 
	 * @param a - complex number
	 * @param b - complex number
	 * @return product of a and b. e.g. a*b
	 */
	public static CNumber multiply(CNumber a, CNumber b) {
		return new CNumber(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
	}

	
	/**
	 * Performs complex division of two numbers.
	 * 
	 * @param a - complex number
	 * @param b - complex number
	 * @return Quotient of a and b. e.g. a/b
	 */
	public static CNumber divide(CNumber a, CNumber b) {
		CNumber quotent = null;

		if (a.equals(ZERO) && !b.equals(ZERO)) {
			quotent = ZERO;
		} else if (b.equals(ZERO)) { // Can not divide by zero
			throw new IllegalArgumentException("Can not divide by zero");
		} else {
			quotent = new CNumber(
					(a.re * b.re + a.im * b.im) / (b.re * b.re + b.im * b.im),

					(a.im * b.re - a.re * b.im) / (b.re * b.re + b.im * b.im));
		}

		return quotent;
	}

	
	/**
	 * Computes absolute value of a number. Note: this will
	 * Always result in a non-negative real number. 
	 * <br>
	 * This method is the same as {@link #mag() mag()}
	 * 
	 * @param value - value to take absolute value of.
	 * @return The absolute value of the number.
	 */
	public static CNumber abs(CNumber value) {
		return new CNumber(value.mag());
	}
	
	
	/**
	 * Computes the multiplicative inverse of a Number. For example, given a number n, the inverse
	 * is 1/n. 
	 * 
	 * This is equivalent to <code> Number.divide(Number.ONE, value); </code>
	 * 
	 * @param value - Number to take inverse of
	 * @return - Inverse of value
	 */
	public static CNumber inv(CNumber value) {
		return CNumber.divide(ONE, value);
	}
	
	
	/**
	 * Computes the additive inverse of a Number. For example, given a number n, the inverse
	 * is -n. 
	 * 
	 * This is equivalent to <code> Number.subtract(Number.ZERO, value); </code>
	 * 
	 * @param value - Number to take inverse of
	 * @return - Inverse of value
	 */
	public static CNumber addInv(CNumber value) {
		return CNumber.subtract(ZERO, value);
	}
	
	
	/**
	 * Performs complex exponentiation of two numbers. Note: <code>base</code> can
	 * be any complex number.
	 * 
	 * TODO: Extend this using to complex numbers using Talyor series expansion of e^x. (May need a new method exponenent()) 
	 * 
	 * @param base - base of exponential expression
	 * @param exponenent - exponent of exponential expression
	 * @return Result of the exponential expression <code>base^exponenent</code>
	 */
	public static CNumber pow(CNumber base, int exponenent) {

		CNumber result;

		if (exponenent == 0) { // Exponent is zero
			result = CNumber.ONE;
		} else if (exponenent > 0) { // Exponent is positive
			result = base;

			for (int i = 1; i < exponenent; i++) {
				result = CNumber.multiply(result, base);
			}
		} else { // Exponent is negative
			result = base;

			for (int i = -1; i > exponenent; i--) {
				result = CNumber.multiply(result, base);
			}

			result = CNumber.divide(CNumber.ONE, result);
		}

		return result;
	}

	
	/**
	 * Performs complex exponentiation of two numbers. Note: The exponent, must be a real integer.
	 * 
	 * TODO: Extend this using to complex numbers using Talyor series expansion of e^x. (May need a new method exp()) 
	 * 
	 * @param base - base of exponential expression
	 * @param exponent - exponent of exponential expression
	 * @return Result of the exponential expression <code>base^exp</code>
	 */
	public static CNumber pow(CNumber base, CNumber exponent) {
		CNumber result = null;

		if (exponent.isComplex()) { // Then we dont have a real number
			throw new IllegalArgumentException("Exponent must be an real number but got " + exponent);
		} else if (exponent.isInt()) { // Then we dont have an integer
			throw new IllegalArgumentException("Exponent must be an integer but got " + exponent);
		} else {
			result = pow(base, Integer.parseInt(exponent.toString()));
		}

		return result;
	}

	
	/**
	 * 
	 * @param x - exponent
	 * @return e^x
	 */
	public static CNumber exp(double x) {
		// TODO: implementation using min max
		return null;
	}
	
	
	/**
	 * 
	 * @param x - exponent
	 * @return e^x
	 */
	public static CNumber exp(CNumber x) {
		// TODO: implementation using min max
		return null;
	}
	
	
	/**
	 * Computes the square root of a number.<br><br>
	 * Note: This method only returns the principle square root 
	 * (the positive square root or square root with positive real part).
	 * <br>If you wish to see all square roots, see {@link #allSqrt(CNumber) allSqrt(CNumber value)}
	 * 
	 * @param value - value to take square root of.
	 * @return Returns the square root of value.
	 */
	public static CNumber sqrt(CNumber value) {
		CNumber result;
		
		if(value.isReal()) {
			if(value.compareToReal(CNumber.ZERO) >= 0) {
				result = new CNumber(Math.sqrt(value.re));
			} else { // Then the number is negative
				result = new CNumber(0, Math.sqrt(-1*value.re));
			}
		} else { // Then the number is complex.
			CNumber r = new CNumber(value.mag());
			
			result = CNumber.multiply(
						sqrt(r), 
						CNumber.divide(
							CNumber.add(value, r),
							new CNumber(CNumber.add(value, r).mag())
						)
					);
			if(result.isReal() && result.compareToReal(CNumber.ZERO) < 0) {
				throw new ArithmeticException("No principle root exists for the complex number " + value);
			}
		}
		
		return result;
	}

	
	/**
	 * Computes all possible square roots of a number.
	 * If you would only like the principle square root then see {@link #sqrt(CNumber) sqrt(CNumber value)} 
	 * 
	 * @param value - value to take square root of.
	 * @return Returns all square roots of value (both positive and negative) in an array.
	 */
	public static CNumber[] allSqrt(CNumber value) {
		CNumber[] result = new CNumber[2];
		result[0] = CNumber.sqrt(value);
		result[1] = CNumber.multiply(CNumber.NEGATIVE_ONE, result[0]);
		return result;
	}
	
	
	/**
	 * The complex signum function. 
	 * 
	 * @param value - Value to evaluet in sign function.
	 * @return If the number is zero then this function returns zero. Otherwise, returns the number divided by its magnitude.
	 */
	public static CNumber sign(CNumber value) {
		
		if(value.equals(CNumber.ZERO)) {
			return CNumber.ZERO;
		} else {
			return CNumber.divide(value, new CNumber(value.mag()));
		}
	}
	
	
	/**
	 * Checks for equivalence between two complex numbers
	 * 
	 * @param b - Number to check equivalence to
	 * @return True if the two numbers are equivalent. Otherwise, returns false.
	 */
	public boolean equals(Object b) {
		boolean result = false;
		CNumber c = (CNumber) b;

		if (this.re == c.re && this.im == c.im) { // Then the two numbers are equal
			result = true;
		}

		return result;
	}
	
	
	/**
	 * Checks for equivalence between two complex numbers
	 * <br><br>
	 * Also see {@link #equals(Object)}
	 * 
	 * @param b - Number to check equivalence to
	 * @return True if the two numbers are equivalent. Otherwise, returns false.
	 */
	public boolean equals(double b) {
		boolean result = false;

		if (this.re==b && this.im==0) { // Then the two numbers are equal
			result = true;
		}

		return result;
	}
	
	
	/**
	 * Rounds number to specified number of decimal places. If the number is complex,
	 * both the real and imaginary parts will be rounded.
	 * 
	 * @param n - Number to round.
	 * @param decimals - Number of decimals to round to.
	 * @return The number <code>n</code> rounded to the specified
	 * 		number of decimals.
	 */
	public static CNumber round(CNumber n, int decimals) {
		
		if (decimals < 0) {
			throw new IllegalArgumentException("Number of decimals must be non-negative but got " +
					decimals);
		}
		
	    double real = BigDecimal.valueOf(n.re).setScale(decimals, RoundingMode.HALF_UP).doubleValue();
	    double imaginary = BigDecimal.valueOf(n.im).setScale(decimals, RoundingMode.HALF_UP).doubleValue();
	    
	    return new CNumber(real, imaginary);
	}
	
	
	/**
	 * Checks if a number is near zero.
	 * 
	 * @param tol - tolerance of how close to zero is
	 * 		considered "near".
	 * @return Returns true if magnitude of number is less than or equal to 
	 * 		<code>tol</code> of zero. Otherwise, returns false.
	 * 
	 */
	public boolean nearZero(double tol) {
		return this.mag() <= tol;
	}
	
	
	/**
	 * Compares the size of two complex numbers (magnitudes). 
	 * 
	 * @param b - Number to compare
	 * @return
	 * - If the magnitude of this number is equal to that of <code>b</code>, then this method will return 0. <br>
	 * - If the magnitude of this number is less than that of <code>b</code>, then this method will return a negative number. <br>
	 * - If the magnitude of this number is greater than that of <code>b</code>, then this method will return a positive number.
	 */
	public int compareTo(CNumber b) {
		if(this.mag() == b.mag()) {return 0;}
		else if(this.mag() < b.mag()) {return -1;}
		else {return 1;}
	}
	
	
	/**
	 * Compares the real value of two numbers. 
	 * 
	 * @param b - Number to compare
	 * @return
	 * - If the real value of this number is equal to that of <code>b</code>, then this method will return 0. <br>
	 * - If the real value of this number is less than that of <code>b</code>, then this method will return a negative number. <br>
	 * - If the real value of this number is greator than that of <code>b</code>, then this method will return a positive number.
	 */
	public int compareToReal(CNumber b) {
		if(this.re == b.re) {return 0;}
		else if(this.re < b.re) {return -1;}
		else {return 1;}
	}
	
	
	/**
	 * Compares the real value of two numbers. 
	 * 
	 * @param b - Number to compare
	 * @return
	 * - If the real value of this number is equal to that of <code>b</code>, then this method will return 0. <br>
	 * - If the real value of this number is less than that of <code>b</code>, then this method will return a negative number. <br>
	 * - If the real value of this number is greater than that of <code>b</code>, then this method will return a positive number.
	 */
	public int compareToReal(double b) {
		return this.compareToReal(new CNumber(b));
	}

	
	/**
	 * Computes the magnitude of a complex number (The numbers distance from the
	 * origin in the complex plane). <br>
	 * This method is the same as {@link #abs(CNumber)}
	 * 
	 * @return magnitude of complex number
	 */
	// TODO: refactor so that the return type is CNumber to make arithmetic with magnitudes easier. Also, add magAsDouble()
	public double mag() {
		return Math.sqrt(this.re * this.re + this.im * this.im);
	}

	
	/**
	 * Computes value with greatest magnitude amongst all passed values. 
	 * 
	 * @param values - a set of values of which to compute the maximum. Each item may be passed as its own variable or as an array.
	 * 	For example, given the CNumbers a, b, c and an array of CNumbers vals, this method may be called in either of the following ways.<br>
	 * 	- <code>CNumber.max(a, b, c)</code><br>
	 *  - <code>CNumber.max(vals)</code>
	 *   
	 * @return Returns value with maximum magnitude amongst values. If two values tie for largest, the first encountered will be returned.
	 */
	public static CNumber max(CNumber... values) {
		CNumber currMax = CNumber.MIN_VALUE;
		
		for(CNumber val : values) {
			if(val.mag() > currMax.mag()) {
				currMax = val;
			}
		}
		
		return currMax;
	}
	
	
	/**
	 * Computes value with smallest magnitude amongst all passed values.
	 * 
	 * @param values - a set of values of which to compute the minimum. Each item may be passed as its own variable or as an array.
	 * 	For example, given the CNumbers a, b, c and an array of CNumbers vals, this method may be called in either of the following ways.<br>
	 * 	- <code>CNumber.max(a, b, c)</code><br>
	 *  - <code>CNumber.max(vals)</code>
	 * 
	 * @return Returns value with minimum magnitude amounts values. If two values tie for smallest, the first encountered will be returned.
	 */
	public static CNumber min(CNumber... values) {
		CNumber currMin = CNumber.MAX_VALUE;
		
		for(CNumber val : values) {
			if(val.mag() < currMin.mag()) {
				currMin = val;
			}
		}
		
		return currMin;
	}
	
	
	/**
	 * @return Real part of number
	 */
	public double getReal() {
		return re;
	}

	
	/**
	 * @return Complex part of number
	 */
	public double getImaginary() {
		return im;
	}

	
	/**
	 * Checks if given number has no imaginary part.
	 * 
	 * @return true if given Number has no imaginary part. Otherwise, returns false.
	 */
	public boolean isReal() {
		return im == 0;
	}
	
	
	/**
	 * Checks if given number has non-zero imaginary part.
	 * 
	 * @return true if given Number has non-zero imaginary part. Otherwise, returns false.
	 */
	public boolean isComplex() {
		return im != 0;
	}
	
	
	/**
	 * Checks if given number has no real part.
	 * 
	 * @return true if given Number has no real part and non-zero imaginary part. Otherwise, returns false.
	 */
	public boolean isImaginary() {
		return re==0 && im != 0;
	}
	
	
	/**
	 * Checks if a give number is a real integer.
	 * 
	 * @return true if number is a real integer. Otherwise, returns false.
	 */
	public boolean isInt() {
		boolean result = true;

		if (this.isComplex()) { // Then we have a complex number.
			result = false;
		} else if (Double.parseDouble(this.toString()) == Math.floor(Double.parseDouble(this.toString()))) { 
			// Then we have a real non-integer.
			result = false;
		}

		return result;
	}

	
	/**
	 * @return real part of given Number as double
	 */
	public double doubleValue() {
		return re;
	}
	

	/**
	 * Note: This method may result in loss of accuracy
	 * 
	 * @return real part of given Number as float
	 */
	public float floatValue() {
		return (float) re;
	}
	

	/**
	 * Note: This method may result in loss of accuracy
	 * 
	 * @return real part of given Number as int
	 */
	public int intValue() {
		return (int) re;
	}

	
	/**
	 * Note: This method may result in loss of accuracy
	 * 
	 * @return real part of given Number as double
	 */
	public long longValue() {
		return (long) re;
	}

	
	/**
	 * @return imaginary part of given Number as double
	 */
	public double doubleImaginaryValue() {
		return im;
	}

	
	/**
	 * Note: This method may result in loss of accuracy
	 * 
	 * @return imaginary part of given Number as float
	 */
	public float floatImaginaryValue() {
		return (float) im;
	}

	/**
	 * Note: This method may result in loss of accuracy
	 * 
	 * @return imaginary part of given Number as int
	 */
	public int intImaginaryValue() {
		return (int) im;
	}

	
	/**
	 * Note: This method may result in loss of accuracy
	 * 
	 * @return imaginary part of given Number as long
	 */
	public long longImaginaryValue() {
		return (long) im;
	}

	
	/**
	 * @param a - a number
	 * @return The length of the string representation of the number
	 */
	public static int length(CNumber a) {
		return a.toString().length();
	}
	
	
	public String toString() {
		String result = "";

		double real = this.re, imaginary = this.im;

		if (real != 0) {
			if (real % 1 == 0) {
				result += (int) real;
			} else {
				result += real;
			}
		}

		if (imaginary != 0) {
			if (imaginary < 0 && real != 0) {
				result += " - ";
				imaginary = -imaginary;
			} else if (real != 0) {
				result += " + ";
			}

			if (imaginary % 1 == 0) {
				if(imaginary != 1) {
					if(imaginary == -1) {
						result += '-';
					} else {
						result += (int) imaginary;
					}
				}			
			} else {
				result += imaginary;
			}

			result += "i";
		}

		if (real == 0 && imaginary == 0) {
			result = "0";
		}

		return result;
	}

	
	// FOR DEVELOPMENT TESTING ONLY //
	public static void main(String[] args) {
		CNumber y = new CNumber("3+2i");
		CNumber x = new CNumber("3+2i");

		System.out.println("\n\n" + x.equals(y) + "\n\n");
	}
}
