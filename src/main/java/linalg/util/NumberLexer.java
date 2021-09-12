package linalg.util;

/**
 * This class produces tokens for a complex number.
 * 
 * There are four tokens, opp, num, im, and eof.
 * 
 * opp is an operator e.g. + or -.
 * num is A standard number
 * im is the imaginary unit i
 * eof is the token representing the end of the number
 * 
 * @since 1.0
 * @author Jacob Watters
 * @version 1.0
 */

/* TODO: This can not be a public class
 */
public class NumberLexer extends Lexer {
	
	/**
	 * @param content - String representation of complex number
	 */
	public NumberLexer(String content) {
		this.content = content;
	}
	
	
	/**
	 * Produces next token from complex number string. Also removes this
	 * token from the string
	 * 
	 * @return Next token in string
	 */
	public Token getNextToken() {	
		
		int state = 1;  // state of FA
        String data = "";  // specific info for the token
        boolean done = false;  
        int sym;  // holds current symbol
        
        
        do {
    		sym = getNextSymbol(); // Will return -1 if there is no symbol to get
        	
        	if(state == 1) {
        		if(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32) { // Whitespace
        			state = 1;
        		}
        		else if(sym == 45) {
        			state = 2;
        			data += (char) sym;
        		} 
        		else if(sym == 43) {
        			state = 4;
        			data += (char) sym;
        			done = true;
        		} 
        		else if(digit(sym)) {
        			state = 3;
        			data += (char) sym;
        		} 
        		else if(sym == 'i') {
        			state = 6;
        			data += (char) sym;
        			done = true;
        		}
        		else if(sym == -1) { // We have reached the end of the string
        			state = 5;
        			done = true;
        		}
        		else {
        			error("Unexpected symbol: " + (char) sym);
        		}
        	}
        	
        	else if(state == 2) {
        		if(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32) { // Whitespace
        			putBackSymbol(sym);
        			done = true;
        		}
        		else if(digit(sym)) {
        			state = 3;	
        			data += (char) sym;
        		} 
        		else {
        			putBackSymbol(sym);
        			done = true;
        		}
        	}
        	else if(state == 3) {
        		if(digit(sym)) {
        			state = 3;	
        			data += (char) sym;
        		} 
        		else if(sym == '.') {
        			state = 7;
        			data += (char) sym;
        		}
        		else {
        			putBackSymbol(sym);
        			done = true;
        		}
        	}
        	else if(state == 7) {
        		if(digit(sym)) {
        			state = 8;
        			data += (char) sym;
        		}
        		else {
        			error("Unexpected symbol: " + (char) sym);
        		}
        	}
        	else if(state == 8) {
        		if(digit(sym)) {
        			state = 8;	
        			data += (char) sym;
        		} 
        		else {
        			putBackSymbol(sym);
        			done = true;
        		}
        	}
        	
        } while(!done);
        
		if(state == 2 || state == 4) { // we have an operator
			return new Token("opp", data);
		}
		else if(state == 3 || state == 8) { // we have a number
			return new Token("num", data);
		}
		else if(state == 5) { // end of number
			return new Token("eof", data);
		}
		else if(state == 6) { // we have the imaginary unit
			return new Token("im", data);
		}
		else {
			error("somehow Lexer FA halted in bad state " + state );
	        return null;
		}
	}
	
	
	// FOR DEVELOPMENT TESTING ONLY //
	public static void main(String[] args) {
		NumberLexer lex = new NumberLexer("1-2i");
		
		while(!lex.content.equals("")) {
			System.out.println(lex.getNextToken());
		}
		
		System.out.println(lex.getNextToken());
	}
}
