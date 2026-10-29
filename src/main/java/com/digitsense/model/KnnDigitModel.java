package com.digitsense.model;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/** k-Nearest Neighbors classifier operating on raw 784-pixel feature vectors. */
public class KnnDigitModel implements DigitModel {

    private final double[][] trainFeatures;
    private final int[] trainLabels;
    private final int k;
    private final String modelVersion;

    public KnnDigitModel(List<double[]> trainFeatures, List<Integer> trainLabels,
                         int k, String modelVersion) {
        this.trainFeatures = trainFeatures.toArray(new double[0][]);
        this.trainLabels = trainLabels.stream().mapToInt(Integer::intValue).toArray();
        this.k = k;
        this.modelVersion = modelVersion;
    }

    @Override
    public PredictionResult predict(BufferedImage image) {
        double[] query = extractFeatures(image);

        // Find k nearest neighbours by Euclidean distance.
        int[] neighbourIndices = new int[k];
        double[] neighbourDistances = new double[k];
        java.util.Arrays.fill(neighbourDistances, Double.MAX_VALUE);

        for (int i = 0; i < trainFeatures.length; i++) {
            double distance = euclideanDistance(query, trainFeatures[i]);
            int worstSlot = 0;
            for (int slot = 1; slot < k; slot++) {
                if (neighbourDistances[slot] > neighbourDistances[worstSlot]) {
                    worstSlot = slot;
                }
            }
            if (distance < neighbourDistances[worstSlot]) {
                neighbourIndices[worstSlot] = i;
                neighbourDistances[worstSlot] = distance;
            }
        }

        // Vote.
        int[] votes = new int[10];
        for (int slot = 0; slot < k; slot++) {
            votes[trainLabels[neighbourIndices[slot]]]++;
        }

        // Find top-3.
        int[] topDigits = new int[3];
        double[] topConfidences = new double[3];
        boolean[] selected = new boolean[10];
        for (int rank = 0; rank < 3; rank++) {
            int best = 0;
            for (int d = 1; d < 10; d++) {
                if (!selected[d] && votes[d] > votes[best]) {
                    best = d;
                }
                if (!selected[d] && votes[d] == votes[best] && !selected[best]) {
                    // tie-break: keep lower digit
                }
            }
            // Skip if the best selected candidate has zero votes and we already picked it.
            if (selected[best]) {
                for (int d = 0; d < 10; d++) {
                    if (!selected[d]) {
                        best = d;
                        break;
                    }
                }
            }
            selected[best] = true;
            topDigits[rank] = best;
            topConfidences[rank] = (double) votes[best] / k;
        }

        return new PredictionResult(topDigits[0], topConfidences[0], modelVersion,
                topDigits, topConfidences);
    }

    /** Converts a 28x28 grayscale image to a 784-element feature vector. */
    public static double[] extractFeatures(BufferedImage image) {
        double[] features = new double[784];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xff;
                int green = (rgb >> 8) & 0xff;
                int blue = rgb & 0xff;
                double gray = (red + green + blue) / (255.0 * 3.0);
                features[y * 28 + x] = 1.0 - gray;
            }
        }
        return features;
    }

    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}

