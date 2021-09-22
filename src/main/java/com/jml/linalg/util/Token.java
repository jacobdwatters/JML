package com.jml.linalg.util;


/* TODO: This can not be a public class
 */
class Token {
	private String kind;
	private String details;
	  
	public Token( String k, String d ) {
		kind = k;  details = d;
	}
	
	
	/**
	 * Checks if token is of kind s
	 * 
	 * @param s - token of interest
	 * @return true if token kind equals s
	 */
	public boolean isKind( String s ) {
	    return kind.equals( s );
	}
	
	
	/**
	 * @return token kind
	 */
	public String getKind() { return kind; }
	
	
	/**
	 * @return token details 
	 */
	public String getDetails() { return details; }
	
	
	/**
	 * Checks if given tokens kind and details match k and d respectivly
	 * 
	 * @param k - token kind
	 * @param d - token details
	 * @return True if token matches kind and details, otherwise false
	 */
	public boolean matches( String k, String d ) {
	    return kind.equals(k) && details.equals(d);
	}
	
	
	/**
	 * If a given tokens kind and details match k and d respectivly then the program will halt
	 * 
	 * @param k - token kind
	 * @param d - token details
	 */
	public void errorCheck(String k, String d) {
		if(!this.matches(k, d)) {
			System.out.println("Expecting token [" + k + "," + d + "] but got " + this);
			System.exit(1);
		}
	}

	
	/**
	 * If a given token does not match the provided kind then the program will halt
	 * 
	 * @param k - token kind
	 */
	public void errorCheck(String k) {
		if(!this.kind.equals(k)) {
			System.out.println("Expecting token of kind " + k + " but got " + this);
			System.exit(1);
		}
	}
	
	
	public String toString() {  
	    return "[" + kind + "," + details + "]";
	}

}
