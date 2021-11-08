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
            throw new IllegalArgumentException("The first tag in the file is not MODEL_TYPE.");
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
            // TODO:
        } else {
            throw new IllegalArgumentException("The file does not seem to contain a valid model.");
        }

        return model;
    }


    /**
     * A helper method which gets content from block.
     *
     * @return The content from the block
     */
    private static String getContent(String block) {
        int start=-1, end=-1;

        for(int i=0; i<block.length(); i++) {
            if(block.charAt(i)=='>' && end==-1) {
                start=i+1;
            } else if(block.charAt(i)=='<' && start>-1) {
                end=i;
                break;
            }
        }

        return block.substring(start, end).replaceAll("\\s","");
    }


    /**
     * A helper method which gets Tag from block.
     *
     * @return The Tag from the block
     */
    private static String getTag(String block) {
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
