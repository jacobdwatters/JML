package linalg.util;


/**
 * Parser for complex number strings or matrix shape strings.
 */
public class Parser {
	
	/**
	 * Parses Matrix shape in the form of 'mxn' where m is the number of rows and n is the number of columns.
	 * 
	 * @param shape - dimensions of matrix (rows by columns) in the form of a string.
	 * @return A double array of length two containing the number of rows and columns respectively
	 */
	public static int[] parseShape(String shape) {
		int[] result = new int[2];
		
		ShapeLexer lex = new ShapeLexer(shape);
		
		Token rows = lex.getNextToken();
		rows.errorCheck("integer");
		
		Token token = lex.getNextToken();
		token.errorCheck("x");

		Token columns = lex.getNextToken();
		columns.errorCheck("integer");
		
		result[0] = Integer.parseInt(rows.getDetails());
		result[1] = Integer.parseInt(columns.getDetails());
		
		return result;
	}
	
	
	/**
	 * Parses a complex number in the form of a string into its real and imaginary parts.
	 * For example, the string <code>"2+3i"</code> would be parsed into real and imaginary parts
	 * <code>2</code> and <code>3</code> respectivly
	 * 
	 * @param num - complex number in one of three forms: <code>a + bi, a,</code> or <code>bi</code> where a and b are
	 * 				real numbers and i is the imaginary unit sqrt(-1)
	 * @return A double array of length two containing the real and imaginary parts of num respectively
	 */
	public static double[] parseNumber(String num) {
		double[] result = new double[2];
		
		NumberLexer lex = new NumberLexer(num);
		
		Token token;
		Token opperator;
		Token real;
		Token imaginary;
		
		token = lex.getNextToken();
		if(token.matches("im", "i")) { // then we have the imaginary unit
			result[0] = 0;
			result[1] = 1;
		}
		else {
			real = token;
			token = lex.getNextToken();
			
			if(token.matches("eof", "")) { // Then we have a real number (a)
				result[0] = Double.parseDouble(real.getDetails());
				result[1] = 0;
			}
			else if(token.matches("im", "i")) { // Then we have a pure imaginary number (bi)
				imaginary = real;
				result[0] = 0;
				
				token = lex.getNextToken();
				token.errorCheck("eof", "");
				
				if(imaginary.getDetails().matches("-")) {
					result[1] = -1;
				}
				else {
					result[1] = Double.parseDouble(imaginary.getDetails());
				}
			}
			else { // Then we have a complex number with nonzero real and imaginary parts (a + bi)
				opperator = token;
				
				if(!opperator.isKind("opp") && opperator.isKind("num")) {
					imaginary = opperator;
				}
				else {
					imaginary = lex.getNextToken();	
				}
				
				if(imaginary.matches("im", "i")) { // Then we have the unit imaginary number
					token = lex.getNextToken();
					token.errorCheck("eof", "");
					
					result[0] = Double.parseDouble(real.getDetails());
					result[1] = 1;
					
					if(opperator.getDetails().equals("-")) { // The operator is negative
						result[1] = -result[1];
					}
				}
				else { // Then we have a multiple of the unit imaginary number
					imaginary.errorCheck("num");
					
					token = lex.getNextToken();
					token.errorCheck("im", "i");
					
					token = lex.getNextToken();
					token.errorCheck("eof", "");
					
					result[0] = Double.parseDouble(real.getDetails());
					result[1] = Double.parseDouble(imaginary.getDetails());
					
					if(opperator.getDetails().equals("-")) { // The operator is negative
						result[1] = -result[1];
					}
				}				
			}
		}
	
		return result;
	}
}
