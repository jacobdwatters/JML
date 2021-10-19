package com.jml.core;


/**
 * Blocks are used to define a model. They can be converted to a markup style equivalent to
 * <pre>
 *     &lt<l></l>name&gt
 *         content
 *     &lt/name&gt
 * </pre>
 *
 * This is useful for saving models in a text format.
 */
public class Block {
    private final String name;
    private final String content;
    private final StringBuilder tagAsString = new StringBuilder();


    /**
     * Constructs a block with specified name and content.
     *
     * @param name Name of the block.
     * @param content Content of the block.
     */
    public Block(String name, String content) {
        this.name = name;
        this.content = content;
        buildBlock();
    }


    // Builds the string format for the tag block
    private void buildBlock() {
        tagAsString.append("<");
        tagAsString.append(name);
        tagAsString.append(">\n");
        tagAsString.append(content);
        tagAsString.append("\n<\\");
        tagAsString.append(name);
        tagAsString.append(">\n");
    }


    /**
     * Gets the markup format of the block.
     *
     * @return Returns the block as a string.
     */
    public String getBlock() {
        return tagAsString.toString();
    }


    /**
     * Builds the content of a markup file containing blocks.
     *
     * @param blocks Blocks to form file from.
     * @return The file contents as a string.
     */
    public static String buildFileContent(Block... blocks) {
        StringBuilder fileContent = new StringBuilder();

        for(Block blk : blocks) {
            fileContent.append(blk.getBlock());
        }

        return fileContent.toString();
    }
}
