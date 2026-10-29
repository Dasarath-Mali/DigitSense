package com.digitsense.ui;

import com.digitsense.model.ModelBenchmark;
import com.digitsense.model.ModelMetrics;

import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Label;
import java.awt.Panel;
import java.util.ArrayList;
import java.util.List;

/** Panel that trains, evaluates, and compares all six digit recognition models. */
public class CompareModelsPanel extends Panel {

    private final Button runBenchmarkButton;
    private final Button viewConfusionButton;
    private final Button viewPerClassButton;
    private final Label progressLabel;
    private final Label headerRow;
    private final java.awt.List resultsList;
    private final Label selectedModelLabel;
    private List<ModelMetrics> benchmarkResults = new ArrayList<>();
    private final Frame parentFrame;

    public CompareModelsPanel(Frame parentFrame) {
        this.parentFrame = parentFrame;
        setLayout(new BorderLayout(6, 6));
        setBackground(new Color(235, 240, 248));

        // ── Header ──
        Panel headerPanel = new Panel(new BorderLayout());
        headerPanel.setBackground(new Color(31, 52, 81));
        Label titleLabel = new Label("  Compare Models", Label.LEFT);
        titleLabel.setForeground(Color.WHITE);
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 15));
        runBenchmarkButton = new Button("Run Benchmark");
        runBenchmarkButton.setBackground(new Color(39, 116, 172));
        runBenchmarkButton.setForeground(Color.WHITE);
        runBenchmarkButton.setFont(new Font("SansSerif", Font.BOLD, 12));
        Panel btnWrap = new Panel(new FlowLayout(FlowLayout.RIGHT));
        btnWrap.setBackground(new Color(31, 52, 81));
        btnWrap.add(runBenchmarkButton);
        headerPanel.add(titleLabel, BorderLayout.CENTER);
        headerPanel.add(btnWrap, BorderLayout.EAST);

        // ── Progress ──
        progressLabel = new Label("Click 'Run Benchmark' to train & evaluate all models on MNIST");
        progressLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        progressLabel.setBackground(new Color(218, 228, 241));

        // ── Column headers ──
        headerRow = new Label(
                "  Model                   Accuracy  Precision   Recall     F1      Time");
        headerRow.setFont(new Font("Monospaced", Font.BOLD, 12));
        headerRow.setBackground(new Color(200, 212, 228));

        // ── Results list ──
        resultsList = new java.awt.List(8, false);
        resultsList.setFont(new Font("Monospaced", Font.PLAIN, 12));
        resultsList.setBackground(Color.WHITE);

        // ── Bottom buttons ──
        viewConfusionButton = new Button("View Confusion Matrix");
        viewConfusionButton.setBackground(new Color(104, 83, 153));
        viewConfusionButton.setForeground(Color.WHITE);
        viewConfusionButton.setEnabled(false);

        viewPerClassButton = new Button("Per-Class Metrics");
        viewPerClassButton.setBackground(new Color(48, 137, 94));
        viewPerClassButton.setForeground(Color.WHITE);
        viewPerClassButton.setEnabled(false);

        selectedModelLabel = new Label("Select a model from the list above");
        selectedModelLabel.setFont(new Font("SansSerif", Font.ITALIC, 11));

        Panel bottomPanel = new Panel(new FlowLayout(FlowLayout.LEFT));
        bottomPanel.setBackground(new Color(218, 228, 241));
        bottomPanel.add(viewConfusionButton);
        bottomPanel.add(viewPerClassButton);
        bottomPanel.add(selectedModelLabel);

        // ── Content panel ──
        Panel contentPanel = new Panel(new BorderLayout(4, 4));
        contentPanel.setBackground(new Color(235, 240, 248));
        contentPanel.add(headerRow, BorderLayout.NORTH);
        contentPanel.add(resultsList, BorderLayout.CENTER);
        contentPanel.add(bottomPanel, BorderLayout.SOUTH);

        // ── Assemble ──
        Panel topBlock = new Panel(new BorderLayout());
        topBlock.add(headerPanel, BorderLayout.NORTH);
        topBlock.add(progressLabel, BorderLayout.SOUTH);

        add(topBlock, BorderLayout.NORTH);
        add(contentPanel, BorderLayout.CENTER);

        registerEvents();
    }

    private void registerEvents() {
        runBenchmarkButton.addActionListener(event -> startBenchmark());

        viewConfusionButton.addActionListener(event -> {
            int selected = resultsList.getSelectedIndex();
            if (selected >= 0 && selected < benchmarkResults.size()) {
                ModelMetrics metrics = benchmarkResults.get(selected);
                ConfusionMatrixDialog dialog = new ConfusionMatrixDialog(
                        parentFrame, metrics.getModelName(), metrics.getConfusionMatrix());
                dialog.setVisible(true);
            }
        });

        viewPerClassButton.addActionListener(event -> {
            int selected = resultsList.getSelectedIndex();
            if (selected >= 0 && selected < benchmarkResults.size()) {
                ModelMetrics metrics = benchmarkResults.get(selected);
                PerClassMetricsDialog dialog = new PerClassMetricsDialog(
                        parentFrame, metrics);
                dialog.setVisible(true);
            }
        });

        resultsList.addItemListener(event -> {
            int selected = resultsList.getSelectedIndex();
            if (selected >= 0 && selected < benchmarkResults.size()) {
                selectedModelLabel.setText("Selected: " + benchmarkResults.get(selected).getModelName());
                viewConfusionButton.setEnabled(true);
                viewPerClassButton.setEnabled(true);
            }
        });
    }

    private void startBenchmark() {
        runBenchmarkButton.setEnabled(false);
        resultsList.removeAll();
        benchmarkResults.clear();
        viewConfusionButton.setEnabled(false);
        viewPerClassButton.setEnabled(false);
        progressLabel.setText("Starting benchmark...");

        Thread benchmarkThread = new Thread(() -> {
            try {
                List<ModelMetrics> results = ModelBenchmark.runBenchmark(
                        (modelName, status, percentComplete) ->
                                EventQueue.invokeLater(() ->
                                        progressLabel.setText(String.format(
                                                "[%d%%] %s - %s",
                                                percentComplete, modelName, status))));

                EventQueue.invokeLater(() -> {
                    benchmarkResults = results;
                    resultsList.removeAll();
                    for (ModelMetrics metrics : results) {
                        resultsList.add(metrics.toTableRow());
                    }
                    progressLabel.setText("Benchmark complete! " + results.size()
                            + " models evaluated. Select a row to view details.");
                    runBenchmarkButton.setEnabled(true);
                });
            } catch (Exception exception) {
                EventQueue.invokeLater(() -> {
                    progressLabel.setText("Benchmark failed: " + exception.getMessage());
                    runBenchmarkButton.setEnabled(true);
                });
                exception.printStackTrace();
            }
        }, "digitsense-benchmark-worker");
        benchmarkThread.setDaemon(true);
        benchmarkThread.start();
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(600, 300);
    }
}

