package linalg.util;

/* TODO: This can not be a public class
 */
class ShapeLexer extends Lexer {
	
	/**
	 * @param content - String representation of matrix shape e.g. "numRows x numCols"
	 */
	public ShapeLexer(String content) {
		this.content = content;
	}
	
	
	/**
	 * @return content of Lexer
	 */
	public String getContent() { return content; };
	
	
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
    		sym = getNextSymbol();
        	
        	if(state == 1) {
        		if(sym == 9 || sym == 10 || sym == 13 ||
                        sym == 32) { // Symbol is Whitespace
        			state = 1; // Do nothing
        		}
        		else if(digit(sym)) {
        			state = 2;
        			data += (char) sym;
        		}
        		else if(sym == 'x' || sym == 'X') {
        			state = 3;
        			done = true; // We are finished with this token
        			data += 'x';
        		}
        		else if(sym == -1) {
        			state = 4;
        			done = true;  			
        		}
        		else { // Content is not in correct format
        			error("Unexpected symbol: " + (char) sym + "\nExpecting a digit");
        		}
        	}        	
        	else if(state == 2) {
        		if(digit(sym)) {
        			state = 2; // Do not change states
        			data += (char) sym;
        		} 
        		else { // We are finished with this token
        			putBackSymbol(sym);
        			done = true;
        		}
        	}
        	// No need for state 3 or 4 as they are excepting states with no outgoing edges and are handeled in state 1
        } while(!done);
        
		if(state == 2) { // we have an integer
			return new Token("integer", data);
		}
		else if(state == 3) { // we have a number
			return new Token("x", data);
		}
		else if(state == 4) {
			return new Token("eof", data);
		}
		else { // The FA has somehow stoped outside of an accept state.
			error("somehow Lexer FA halted in bad state " + state );
	        return null;
		}
	}
}
