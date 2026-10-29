package com.digitsense.model;

import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/** Trains and evaluates all six model types on MNIST, returning their metrics. */
public class ModelBenchmark {

    private static final int BATCH_SIZE = 64;
    private static final int SEED = 12345;
    private static final int ENSEMBLE_SIZE = 10;

    /** Callback for tracking benchmark progress in the UI. */
    public interface ProgressListener {
        void onProgress(String modelName, String status, int percentComplete);
    }

    /** Runs the full benchmark and returns metrics for every model. */
    public static List<ModelMetrics> runBenchmark(ProgressListener listener) throws Exception {
        List<ModelMetrics> results = new ArrayList<>();

        // 1. Logistic Regression
        listener.onProgress("Logistic Regression", "Training...", 0);
        results.add(trainAndEvaluateDl4j("Logistic Regression",
                LogisticRegressionModel.buildNetwork(SEED), 3, true, listener, 0));

        // 2. KNN
        listener.onProgress("KNN (k=3)", "Loading training data...", 17);
        results.add(trainAndEvaluateKnn(listener));

        // 3. Random Forest (ensemble)
        listener.onProgress("Random Forest", "Training ensemble...", 33);
        results.add(trainAndEvaluateRandomForest(listener));

        // 4. MLP
        listener.onProgress("MLP", "Training...", 50);
        results.add(trainAndEvaluateDl4j("MLP",
                MlpDigitModel.buildNetwork(SEED), 3, true, listener, 50));

        // 5. LeNet CNN
        listener.onProgress("LeNet CNN", "Training...", 67);
        results.add(trainAndEvaluateDl4j("LeNet CNN",
                LeNetDigitModel.buildNetwork(SEED), 3, false, listener, 67));

        // 6. Deeper CNN
        listener.onProgress("Deeper CNN", "Training...", 83);
        results.add(trainAndEvaluateDl4j("Deeper CNN",
                DeeperCnnDigitModel.buildNetwork(SEED), 5, false, listener, 83));

        listener.onProgress("All models", "Benchmark complete!", 100);
        return results;
    }

    /** Trains and evaluates a DL4J network-based model. */
    @SuppressWarnings("deprecation")
    private static ModelMetrics trainAndEvaluateDl4j(String modelName,
                                                      MultiLayerNetwork network,
                                                      int epochs, boolean flatten,
                                                      ProgressListener listener,
                                                      int basePercent) throws Exception {
        long startTime = System.currentTimeMillis();

        DataSetIterator trainIter = new MnistDataSetIterator(BATCH_SIZE, true, SEED);
        DataSetIterator testIter = new MnistDataSetIterator(BATCH_SIZE, false, SEED);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            trainIter.reset();
            if (flatten) {
                // For flat-input models, reshape each batch from [N,1,28,28] to [N,784]
                while (trainIter.hasNext()) {
                    DataSet batch = trainIter.next();
                    INDArray features = batch.getFeatures();
                    if (features.rank() == 4) {
                        features = features.reshape(features.size(0), 784);
                        batch.setFeatures(features);
                    }
                    network.fit(batch);
                }
            } else {
                network.fit(trainIter);
            }
            listener.onProgress(modelName,
                    "Epoch " + epoch + "/" + epochs + " complete",
                    basePercent + (epoch * 16 / epochs));
        }

        long trainingTime = System.currentTimeMillis() - startTime;

        // Evaluate
        Evaluation evaluation = new Evaluation(10);
        testIter.reset();
        while (testIter.hasNext()) {
            DataSet batch = testIter.next();
            INDArray features = batch.getFeatures();
            if (flatten && features.rank() == 4) {
                features = features.reshape(features.size(0), 784);
            }
            INDArray output = network.output(features, false);
            evaluation.eval(batch.getLabels(), output);
        }

        return buildMetrics(modelName, evaluation, trainingTime);
    }

    /** Trains and evaluates the KNN model using a subset of MNIST. */
    @SuppressWarnings("deprecation")
    private static ModelMetrics trainAndEvaluateKnn(ProgressListener listener) throws Exception {
        long startTime = System.currentTimeMillis();

        // Load training data (first 10000 samples for performance)
        DataSetIterator trainIter = new MnistDataSetIterator(BATCH_SIZE, true, SEED);
        List<double[]> trainFeatures = new ArrayList<>();
        List<Integer> trainLabels = new ArrayList<>();
        int maxTrain = 10000;

        while (trainIter.hasNext() && trainFeatures.size() < maxTrain) {
            DataSet batch = trainIter.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            if (features.rank() == 4) {
                features = features.reshape(features.size(0), 784);
            }

            for (int i = 0; i < features.size(0) && trainFeatures.size() < maxTrain; i++) {
                double[] featureVector = new double[784];
                for (int j = 0; j < 784; j++) {
                    featureVector[j] = features.getDouble(i, j);
                }
                trainFeatures.add(featureVector);
                trainLabels.add(labels.getRow(i).argMax().getInt(0));
            }
        }

        listener.onProgress("KNN (k=3)", "Data loaded, evaluating...", 22);

        long trainingTime = System.currentTimeMillis() - startTime;

        KnnDigitModel knn = new KnnDigitModel(trainFeatures, trainLabels, 3, "knn-k3");

        // Evaluate on a subset of test data (first 2000 for speed)
        DataSetIterator testIter = new MnistDataSetIterator(BATCH_SIZE, false, SEED);
        int[][] confusionMatrix = new int[10][10];
        int correct = 0;
        int total = 0;
        int maxTest = 2000;

        while (testIter.hasNext() && total < maxTest) {
            DataSet batch = testIter.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            if (features.rank() == 4) {
                features = features.reshape(features.size(0), 784);
            }

            for (int i = 0; i < features.size(0) && total < maxTest; i++) {
                double[] query = new double[784];
                for (int j = 0; j < 784; j++) {
                    query[j] = features.getDouble(i, j);
                }
                int actual = labels.getRow(i).argMax().getInt(0);

                // Find k=3 nearest
                int predicted = predictKnn(query, trainFeatures, trainLabels, 3);
                confusionMatrix[actual][predicted]++;
                if (predicted == actual) correct++;
                total++;
            }
            if (total % 500 == 0) {
                listener.onProgress("KNN (k=3)", "Evaluated " + total + "/" + maxTest, 22);
            }
        }

        return buildMetricsFromConfusion("KNN (k=3)", confusionMatrix, correct, total,
                System.currentTimeMillis() - startTime);
    }

    /** Fast KNN prediction on a raw feature vector. */
    private static int predictKnn(double[] query, List<double[]> trainFeatures,
                                   List<Integer> trainLabels, int k) {
        int[] bestIndices = new int[k];
        double[] bestDistances = new double[k];
        java.util.Arrays.fill(bestDistances, Double.MAX_VALUE);

        for (int i = 0; i < trainFeatures.size(); i++) {
            double dist = 0;
            double[] train = trainFeatures.get(i);
            for (int j = 0; j < 784; j++) {
                double diff = query[j] - train[j];
                dist += diff * diff;
            }
            int worstSlot = 0;
            for (int s = 1; s < k; s++) {
                if (bestDistances[s] > bestDistances[worstSlot]) worstSlot = s;
            }
            if (dist < bestDistances[worstSlot]) {
                bestIndices[worstSlot] = i;
                bestDistances[worstSlot] = dist;
            }
        }

        int[] votes = new int[10];
        for (int s = 0; s < k; s++) {
            votes[trainLabels.get(bestIndices[s])]++;
        }
        int best = 0;
        for (int d = 1; d < 10; d++) {
            if (votes[d] > votes[best]) best = d;
        }
        return best;
    }

    /** Trains and evaluates the Random Forest (ensemble) model. */
    @SuppressWarnings("deprecation")
    private static ModelMetrics trainAndEvaluateRandomForest(
            ProgressListener listener) throws Exception {
        long startTime = System.currentTimeMillis();
        List<MultiLayerNetwork> ensemble = new ArrayList<>();

        for (int member = 0; member < ENSEMBLE_SIZE; member++) {
            MultiLayerNetwork net = RandomForestModel.buildEnsembleMember(SEED + member);
            DataSetIterator trainIter = new MnistDataSetIterator(BATCH_SIZE, true,
                    SEED + member);

            for (int epoch = 0; epoch < 2; epoch++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    DataSet batch = trainIter.next();
                    INDArray features = batch.getFeatures();
                    if (features.rank() == 4) {
                        features = features.reshape(features.size(0), 784);
                        batch.setFeatures(features);
                    }
                    net.fit(batch);
                }
            }
            ensemble.add(net);
            listener.onProgress("Random Forest",
                    "Trained member " + (member + 1) + "/" + ENSEMBLE_SIZE,
                    33 + (member * 16 / ENSEMBLE_SIZE));
        }

        // Evaluate ensemble
        DataSetIterator testIter = new MnistDataSetIterator(BATCH_SIZE, false, SEED);
        Evaluation evaluation = new Evaluation(10);

        while (testIter.hasNext()) {
            DataSet batch = testIter.next();
            INDArray features = batch.getFeatures();
            if (features.rank() == 4) {
                features = features.reshape(features.size(0), 784);
            }

            // Average ensemble outputs
            INDArray sumOutput = null;
            for (MultiLayerNetwork net : ensemble) {
                INDArray output = net.output(features, false);
                if (sumOutput == null) {
                    sumOutput = output.dup();
                } else {
                    sumOutput.addi(output);
                }
            }
            sumOutput.divi(ENSEMBLE_SIZE);
            evaluation.eval(batch.getLabels(), sumOutput);
        }

        return buildMetrics("Random Forest", evaluation,
                System.currentTimeMillis() - startTime);
    }

    /** Builds ModelMetrics from a DL4J Evaluation object. */
    @SuppressWarnings("deprecation")
    private static ModelMetrics buildMetrics(String modelName, Evaluation evaluation,
                                              long trainingTimeMs) {
        double accuracy = evaluation.accuracy();
        double[] precisionPerClass = new double[10];
        double[] recallPerClass = new double[10];
        double[] f1PerClass = new double[10];

        for (int c = 0; c < 10; c++) {
            precisionPerClass[c] = evaluation.precision(c);
            recallPerClass[c] = evaluation.recall(c);
            f1PerClass[c] = evaluation.f1(c);
        }

        double macroPrecision = 0, macroRecall = 0, macroF1 = 0;
        for (int c = 0; c < 10; c++) {
            macroPrecision += precisionPerClass[c];
            macroRecall += recallPerClass[c];
            macroF1 += f1PerClass[c];
        }
        macroPrecision /= 10;
        macroRecall /= 10;
        macroF1 /= 10;

        // Build confusion matrix from evaluation
        int[][] confusionMatrix = new int[10][10];
        try {
            String confStr = evaluation.confusionToString();
            // Parse the confusion matrix from the evaluation stats
            // Fallback: compute from evaluation's internal state
            for (int actual = 0; actual < 10; actual++) {
                for (int predicted = 0; predicted < 10; predicted++) {
                    // Use the confusion matrix from evaluation
                    confusionMatrix[actual][predicted] =
                            evaluation.getConfusionMatrix().getCount(actual, predicted);
                }
            }
        } catch (Exception ignored) {
            // If parsing fails, leave zeros
        }

        return new ModelMetrics(modelName, accuracy, precisionPerClass, recallPerClass,
                f1PerClass, macroPrecision, macroRecall, macroF1,
                confusionMatrix, trainingTimeMs);
    }

    /** Builds ModelMetrics directly from a confusion matrix. */
    private static ModelMetrics buildMetricsFromConfusion(String modelName,
                                                           int[][] confusionMatrix,
                                                           int correct, int total,
                                                           long trainingTimeMs) {
        double accuracy = total > 0 ? (double) correct / total : 0;
        double[] precisionPerClass = new double[10];
        double[] recallPerClass = new double[10];
        double[] f1PerClass = new double[10];

        for (int c = 0; c < 10; c++) {
            int tp = confusionMatrix[c][c];
            int fpSum = 0;
            int fnSum = 0;
            for (int j = 0; j < 10; j++) {
                if (j != c) {
                    fpSum += confusionMatrix[j][c]; // false positives
                    fnSum += confusionMatrix[c][j]; // false negatives
                }
            }
            precisionPerClass[c] = (tp + fpSum) > 0 ? (double) tp / (tp + fpSum) : 0;
            recallPerClass[c] = (tp + fnSum) > 0 ? (double) tp / (tp + fnSum) : 0;
            double p = precisionPerClass[c], r = recallPerClass[c];
            f1PerClass[c] = (p + r) > 0 ? 2.0 * p * r / (p + r) : 0;
        }

        double macroPrecision = 0, macroRecall = 0, macroF1 = 0;
        for (int c = 0; c < 10; c++) {
            macroPrecision += precisionPerClass[c];
            macroRecall += recallPerClass[c];
            macroF1 += f1PerClass[c];
        }
        macroPrecision /= 10;
        macroRecall /= 10;
        macroF1 /= 10;

        return new ModelMetrics(modelName, accuracy, precisionPerClass, recallPerClass,
                f1PerClass, macroPrecision, macroRecall, macroF1,
                confusionMatrix, trainingTimeMs);
    }
}

