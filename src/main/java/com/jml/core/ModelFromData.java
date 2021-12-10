package com.jml.core;

import java.util.ArrayList;
import java.util.List;

class ModelFromData {
    private ModelFromData() {throw new IllegalStateException("Utility class. Cannot be instantiated");}


    // Creates a model from a list of blocks.
    static Model create(List<String> blocks) {
        Model model = null;
        String block;
        List<String> tags = new ArrayList<>(), contents = new ArrayList<>();

        while(!blocks.isEmpty()) {
            block = blocks.remove(0); // get the next block
            tags.add(getTag(block));
            contents.add(getContent(block));
        }

        if(!tags.get(0).equals("MODEL_TYPE")) {
            throw new IllegalArgumentException("Invalid file. The first tag in the file is not MODEL_TYPE.");
        }

        String modelType = contents.remove(0);
        tags.remove(0);

        if(modelType.equals(ModelTypes.POLYNOMIAL_REGRESSION.toString())) {
            model = PolyRegFromData.create(tags, contents);

        } else if(modelType.equals(ModelTypes.LINEAR_REGRESSION.toString())) {
            model = LinRegFromData.create(tags, contents);

        } else if(modelType.equals(ModelTypes.MULTIPLE_LINEAR_REGRESSION.toString())) {
            model = MultRegFromData.create(tags, contents);

        } else if(modelType.equals(ModelTypes.K_NEAREST_NEIGHBORS.toString())){
            model = KnnFromData.create(tags, contents);

        } else if(modelType.equals(ModelTypes.LOGISTIC_REGRESSION.toString())) {
            model = LogRegFromData.create(tags, contents);

        } else if(modelType.equals(ModelTypes.PERCEPTRON.toString())) {
            // TODO:
        } else if(modelType.equals(ModelTypes.NEURAL_NETWORK.toString())) {
            model = NeuralNetFromData.create(tags, contents);
        } else {
            throw new IllegalArgumentException("The file does not seem to contain a valid model.");
        }

        return model;
    }


    /**
     * A helper method which gets content from block. This method assumes the block has already been properly
     * parsed and the passed string only contains a single block. The single block may be a parent block with sub-blocks.
     *
     * @param block Block to get the content of.
     * @return The content from the block
     */
    protected static String getContent(String block) {
        int start=-1, end=-1, endConsideration = -1;
        StringBuilder currentTag;
        String tag = getTag(block);

        for(int i=0; i<block.length(); i++) {
            if(block.charAt(i)=='>' && end==-1 && start==-1) {
                start=i+1;
            } else if(block.charAt(i)=='<' && start>-1) {
                currentTag = new StringBuilder();

                for(int j=i+2; j<block.length(); j++) {
                    if(block.charAt(j) != '>') {
                        currentTag.append(block.charAt(j));
                    } else {
                        break;
                    }
                }

                if(currentTag.toString().equals(tag)) { // Ensure that the ending tag is the same as the opening tag.
                    end=i;
                    break;
                }
            }
        }

        if(start==-1) { // Did not find start of block.
            throw new IllegalStateException("Error while parsing block. Could not find start.");
        }

        if(end==-1) { // Did not find end of block.
            throw new IllegalStateException("Error while parsing block. Could not find end.");
        }

        return block.substring(start, end).replaceAll("\\s","");
    }


    /**
     * A helper method which gets Tag from block. This method assumes the block has already been properly
     * parsed and the passed string only contains a single block. The single block may be a parent block with sub-blocks.
     *
     * @param block Block to get the tag of.
     * @return The Tag from the block
     */
    protected static String getTag(String block) {
        boolean foundStart = false;
        StringBuilder tag = new StringBuilder();

        for(int i=0; i<block.length(); i++) {
            if(block.charAt(i)=='<') {
                foundStart = true;
            } else if(foundStart && block.charAt(i)!='>') {
                tag.append(block.charAt(i)); // Add the character to the tag.
            } else if(block.charAt(i)=='>') {
                break; // Then we are done.
            }
        }

        return tag.toString();
    }
}
