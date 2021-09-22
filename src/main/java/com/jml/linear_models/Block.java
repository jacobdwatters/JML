package com.jml.linear_models;

class Block {
    private String name = "";
    private String content = "";
    private StringBuilder tagAsString = new StringBuilder();


    protected Block(String name, String content) {
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


    // Returns the block as a String.
    protected String getBlock() {
        return tagAsString.toString();
    }


    protected static String buildFileContent(Block... blocks) {
        StringBuilder fileContent = new StringBuilder();

        for(Block blk : blocks) {
            fileContent.append(blk.getBlock());
        }

        return fileContent.toString();
    }
}
