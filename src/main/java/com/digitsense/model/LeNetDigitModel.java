package com.digitsense.model;

import java.awt.image.BufferedImage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * LeNet-style Convolutional Neural Network for digit recognition.
 * Architecture: Conv(5x5,20) -> Pool -> Conv(5x5,50) -> Pool -> Dense(500) -> Output(10).
 */
public class LeNetDigitModel implements DigitModel {

    private final MultiLayerNetwork network;
    private final String modelVersion;

    public LeNetDigitModel(MultiLayerNetwork network, String modelVersion) {
        this.network = network;
        this.modelVersion = modelVersion;
    }

    @Override
    public PredictionResult predict(BufferedImage image) {
        INDArray input = Nd4j.create(1, 1, 28, 28);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xff;
                int green = (rgb >> 8) & 0xff;
                int blue = rgb & 0xff;
                double gray = (red + green + blue) / (255.0 * 3.0);
                input.putScalar(new int[]{0, 0, y, x}, 1.0 - gray);
            }
        }

        INDArray probabilities = network.output(input, false);
        return ModelPredictionHelper.extractTopPredictions(probabilities, modelVersion);
    }

    /** Builds an untrained LeNet-style CNN. */
    public static MultiLayerNetwork buildNetwork(int seed) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .nOut(20)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nOut(50)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        return network;
    }
}

