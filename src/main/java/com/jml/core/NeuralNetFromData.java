package com.jml.core;

import com.jml.neural_network.ModelTags;
import com.jml.neural_network.NeuralNetwork;
import com.jml.neural_network.activations.ActivationFunction;
import com.jml.neural_network.activations.Activations;
import com.jml.neural_network.layers.Dense;
import com.jml.neural_network.layers.Layer;
import linalg.Matrix;

import java.util.*;

class NeuralNetFromData extends NeuralNetwork {

    private static Scanner scanner;

    static Model create(List<String> tags, List<String> contents) {
        NeuralNetFromData neuralNetModel = new NeuralNetFromData();
        String tag, content;

        neuralNetModel.isFit = true; // Since we are loading a pretrained model, set to true.

        while(!tags.isEmpty() && !contents.isEmpty()) { // Parse through all blocks from the file.
            tag = tags.remove(0);
            content = contents.remove(0);
            scanner = new Scanner(content);

            if(tag.equals(ModelTags.HYPER_PARAMETERS.toString())) {
                String[] hypParams = content.split(",");

                if(hypParams.length != 4) { // Ensure that there are four hyper-parameters.
                    throw new IllegalStateException("Failed to load model: Unexpected number of hyper-parameters.");
                }

                neuralNetModel.learningRate = Double.parseDouble(hypParams[0]);
                neuralNetModel.epochs = Integer.parseInt(hypParams[1]);
                neuralNetModel.batchSize = Integer.parseInt(hypParams[2]);
                neuralNetModel.threshold = Double.parseDouble(hypParams[3]);

            } else if(tag.equals(ModelTags.LAYER.toString())) {

                List<String> lines = new ArrayList<String>();
                List<String> blocks = new ArrayList<>();
                List<String> layerTags = new ArrayList<>(), layerContents = new ArrayList<>();

                /* Reformat the layer contents so that the sub-blocks can be parsed.*/
                // TODO: This is a little ridiculous, write a real parser that can deal with sub-blocks properly.
                String[] temp = content.replace(">", ">\n")
                        .replace("<", "\n<")
                        .replace("\n\n", "\n").trim()
                        .split("\n");
                /* End of reformatting layer contents*/

                Collections.addAll(lines, temp);

                // get all blocks of the layer.
                while(!lines.isEmpty()) {
                    blocks.add(Model.nextBlock(lines));
                }

                // Extract all tags and contents from the blocks of this layer.
                for(String blk : blocks) {
                    layerTags.add(ModelFromData.getTag(blk));
                    layerContents.add(ModelFromData.getContent(blk));
                }

                neuralNetModel.add(createLayer(layerTags, layerContents)); // Construct layer and add to model.

            } else if(tag.equals(ModelTags.OPTIMIZER.toString())) {
                // TODO:
            }
            else {
                throw new IllegalArgumentException("Failed to load model: Unrecognized tag in file: " + tag);
            }
        }

        return neuralNetModel;
    }


    private static Layer createLayer(List<String> layerTags, List<String> layerContents) {
        Layer layer = null;
        String type, tag, content;
        ActivationFunction activation = null;
        Matrix weights = null, bias = null;
        int inDim=-1, outDim=-1;

        if(!layerTags.get(0).equals(ModelTags.TYPE.toString())) {
            throw new IllegalStateException("Failed to load model: First tag in neural network layer must be type but got "
                 + layerTags.get(0));
        } else {
            layerTags.remove(0);
            type = layerContents.remove(0);
        }

        while(layerTags.size()!=0 && layerContents.size()!=0) {
            tag = layerTags.remove(0);
            content = layerContents.remove(0);

            if(tag.equals(ModelTags.ACTIVATION.toString())) {
                // Extract the activation function.
                if(content.equalsIgnoreCase("linear")) {
                    // Then we have a linear activation
                    activation = Activations.linear;
                } else if(content.equalsIgnoreCase("sigmoid")) {
                    // Then we have a sigmoid activation
                    activation = Activations.sigmoid;
                } else if(content.equalsIgnoreCase("relu")) {
                    // Then we have a ReLU activation.
                    activation = Activations.relu;
                } else if(content.equalsIgnoreCase("tanh")) {
                    // Then we have a ReLU activation.
                    activation = Activations.tanh;
                } else {
                    throw new IllegalStateException("Unknown activation function: " + content);
                }

            } else if(tag.equals(ModelTags.DIMENSIONS.toString())) {
                // Extract dimensions of the layer.
                String[] dims = content.split(",");
                inDim = Integer.parseInt(dims[0]);
                outDim = Integer.parseInt(dims[1]);

            } else if(tag.equals(ModelTags.WEIGHTS.toString())) {
                // Form the weight matrix for the layer.
                String[] rows = content.split(";");
                int rowLength = rows[0].split(",").length;
                double[][] vals = new double[rows.length][rowLength];

                for(int i=0; i<vals.length; i++) {
                    String[] rowVals = rows[i].split(",");

                    for(int j=0; j<vals[0].length; j++) {
                        vals[i][j] = Double.parseDouble(rowVals[j]);
                    }
                }

                weights = new Matrix(vals);

            } else if(tag.equals(ModelTags.BIAS.toString())) {
                // Form the bias vector for the layer.
                String[] v = content.split(";");
                double[][] vals = new double[v.length][1];

                for(int i=0; i<vals.length; i++) {
                    vals[i][0] = Double.parseDouble(v[i]);
                }

                bias = new Matrix(vals);

            } else {
                throw new IllegalStateException("Unrecognized tag (" + tag + ") encountered for layer.");
            }
        }

        // Create the layer
        if(type.equals("Dense")) {
            layer = new Dense(inDim, outDim, activation);
            layer.setBias(bias);
            layer.setWeights(weights);

        } else {
            throw new IllegalStateException("Unrecognized layer (" + type + ") encountered for neural network.");
        }

        return layer;
    }
}
