package com.digitsense.model;

import org.nd4j.linalg.api.ndarray.INDArray;

/** Shared helper for extracting top-3 predictions from a softmax probability vector. */
final class ModelPredictionHelper {

    private ModelPredictionHelper() {
    }

    /** Extracts top-3 digits and their confidences from a [1, 10] probability vector. */
    static PredictionResult extractTopPredictions(INDArray probabilities, String modelVersion) {
        int[] topDigits = new int[3];
        double[] topConfidences = new double[3];
        boolean[] selected = new boolean[10];

        for (int rank = 0; rank < topDigits.length; rank++) {
            int bestDigit = 0;
            double bestProbability = -1.0;
            for (int candidate = 0; candidate < 10; candidate++) {
                double probability = probabilities.getDouble(0, candidate);
                if (!selected[candidate] && probability > bestProbability) {
                    bestDigit = candidate;
                    bestProbability = probability;
                }
            }
            selected[bestDigit] = true;
            topDigits[rank] = bestDigit;
            topConfidences[rank] = bestProbability;
        }

        return new PredictionResult(topDigits[0], topConfidences[0], modelVersion,
                topDigits, topConfidences);
    }
}

