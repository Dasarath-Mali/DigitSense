package com.digitsense.model;

/** Holds evaluation metrics for a single model. */
public class ModelMetrics {

    private final String modelName;
    private final double accuracy;
    private final double[] precisionPerClass;
    private final double[] recallPerClass;
    private final double[] f1PerClass;
    private final double macroPrecision;
    private final double macroRecall;
    private final double macroF1;
    private final int[][] confusionMatrix;
    private final double trainingTimeMs;

    public ModelMetrics(String modelName, double accuracy,
                        double[] precisionPerClass, double[] recallPerClass,
                        double[] f1PerClass,
                        double macroPrecision, double macroRecall, double macroF1,
                        int[][] confusionMatrix, double trainingTimeMs) {
        this.modelName = modelName;
        this.accuracy = accuracy;
        this.precisionPerClass = precisionPerClass;
        this.recallPerClass = recallPerClass;
        this.f1PerClass = f1PerClass;
        this.macroPrecision = macroPrecision;
        this.macroRecall = macroRecall;
        this.macroF1 = macroF1;
        this.confusionMatrix = confusionMatrix;
        this.trainingTimeMs = trainingTimeMs;
    }

    public String getModelName() { return modelName; }
    public double getAccuracy() { return accuracy; }
    public double[] getPrecisionPerClass() { return precisionPerClass; }
    public double[] getRecallPerClass() { return recallPerClass; }
    public double[] getF1PerClass() { return f1PerClass; }
    public double getMacroPrecision() { return macroPrecision; }
    public double getMacroRecall() { return macroRecall; }
    public double getMacroF1() { return macroF1; }
    public int[][] getConfusionMatrix() { return confusionMatrix; }
    public double getTrainingTimeMs() { return trainingTimeMs; }

    /** Formatted one-line summary for display in a list. */
    public String toTableRow() {
        return String.format("%-22s  %6.2f%%   %6.2f%%    %6.2f%%  %6.2f%%   %6.1fs",
                modelName,
                accuracy * 100, macroPrecision * 100,
                macroRecall * 100, macroF1 * 100,
                trainingTimeMs / 1000.0);
    }
}

