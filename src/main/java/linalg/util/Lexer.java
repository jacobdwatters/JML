package linalg.util;

/**
 * Performs lexicographical analysis and produces tokens.
 */
class Lexer {
	protected String content = ""; // Content of Lexer
	
	
	/**
	 * @return content of Lexer
	 */
	public String getContent() { return content; };
	
	
	/** 
	 * @param code - ascii value of character
	 * @return returns true if character is digit. Otherwise, returns false.
	 */
	protected boolean digit(int code ) {  
	    return 48<=code && code<=57;
	}
	
	
	
	/**
	 * Produces individual symbols from content, left to right, as ascii values. 
	 * 
	 * @return Returns ascii value of the next symbol from content. If content is empty then returns -1
	 */
	protected int getNextSymbol() {
		int result = -1;	
		
		if (content.length() > 0) {
			result = content.charAt(0);
			content = content.substring(1, content.length());
		}
		
		return result;
	}
	
	
	/**
	 * Replaces unneded symbol back into content string.
	 * 
	 * Note: This method should only be used when the programer is confident the token
	 * is not an unexpected token.
	 * 
	 * @param sym - symbol to place back into content string
	 */
	protected void putBackSymbol(int sym) {		
		if(sym == -1) {
			content = "";
		}
		else {
			content = (char) sym + content;
		}
		
	}
	
	
	/**
	 * Stops exicution with an error message
	 * 
	 * @param message - error message to print
	 */
	protected static void error( String message ) {
	     System.out.println( message );
	     System.exit(1);
	}
}
