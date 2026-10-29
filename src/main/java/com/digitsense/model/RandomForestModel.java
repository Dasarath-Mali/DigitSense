package com.digitsense.model;

import java.awt.image.BufferedImage;
import java.util.List;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Random Forest approximation using an ensemble of shallow neural networks.
 * Each member is trained on a different random seed, and predictions are
 * averaged across all members.
 */
public class RandomForestModel implements DigitModel {

    private final List<MultiLayerNetwork> ensemble;
    private final String modelVersion;

    public RandomForestModel(List<MultiLayerNetwork> ensemble, String modelVersion) {
        this.ensemble = ensemble;
        this.modelVersion = modelVersion;
    }

    @Override
    public PredictionResult predict(BufferedImage image) {
        INDArray input = Nd4j.create(1, 784);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xff;
                int green = (rgb >> 8) & 0xff;
                int blue = rgb & 0xff;
                double gray = (red + green + blue) / (255.0 * 3.0);
                input.putScalar(new int[]{0, y * 28 + x}, 1.0 - gray);
            }
        }

        // Average softmax outputs across all ensemble members.
        INDArray sumProbabilities = Nd4j.zeros(1, 10);
        for (MultiLayerNetwork member : ensemble) {
            sumProbabilities.addi(member.output(input, false));
        }
        sumProbabilities.divi(ensemble.size());

        return ModelPredictionHelper.extractTopPredictions(sumProbabilities, modelVersion);
    }

    /** Builds an untrained shallow network (one hidden layer) for the ensemble. */
    public static MultiLayerNetwork buildEnsembleMember(int seed) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(784)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        return network;
    }
}

