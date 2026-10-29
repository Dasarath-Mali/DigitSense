package com.digitsense.model;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/** Selects the trained model when it exists and otherwise keeps the app runnable. */
public final class ModelFactory {

    private static final File MODEL_FILE = new File("models/digitsense-cnn.zip");

    /** All model names that can appear in the Quick Predict dropdown. */
    private static final List<String> MODEL_NAMES = Arrays.asList(
            "CNN (default)",
            "Logistic Regression",
            "MLP",
            "LeNet CNN",
            "Deeper CNN"
    );

    private ModelFactory() {
    }

    public static DigitModel loadAvailableModel() {
        if (!MODEL_FILE.isFile()) {
            return new UntrainedDigitModel();
        }

        try {
            return new Dl4jDigitModel(MODEL_FILE, "digitsense-cnn-v1");
        } catch (IOException | RuntimeException exception) {
            System.err.println("CNN model could not be loaded: " + exception.getMessage());
            return new UntrainedDigitModel();
        }
    }

    /** Returns the names of models available for quick-prediction selection. */
    public static List<String> getAvailableModelNames() {
        return MODEL_NAMES;
    }

    /**
     * Loads a model by its display name.
     * <p>For named variants other than the default CNN, the network is
     * constructed fresh and untrained (the benchmark trains them on MNIST
     * before evaluation). Quick Predict uses the default CNN or an untrained
     * stand-in when no trained file exists.</p>
     */
    public static DigitModel loadModelByName(String name) {
        if (name == null) {
            return loadAvailableModel();
        }
        return switch (name) {
            case "Logistic Regression" -> {
                var net = LogisticRegressionModel.buildNetwork(12345);
                yield new LogisticRegressionModel(net, "logistic-regression");
            }
            case "MLP" -> {
                var net = MlpDigitModel.buildNetwork(12345);
                yield new MlpDigitModel(net, "mlp");
            }
            case "LeNet CNN" -> {
                var net = LeNetDigitModel.buildNetwork(12345);
                yield new LeNetDigitModel(net, "lenet-cnn");
            }
            case "Deeper CNN" -> {
                var net = DeeperCnnDigitModel.buildNetwork(12345);
                yield new DeeperCnnDigitModel(net, "deeper-cnn");
            }
            default -> loadAvailableModel();
        };
    }
}
