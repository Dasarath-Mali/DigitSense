package com.digitsense.ui;

import com.digitsense.image.ImagePreprocessor;
import com.digitsense.model.DigitModel;
import com.digitsense.model.ModelFactory;
import com.digitsense.model.PredictionResult;

import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Choice;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Label;
import java.awt.Panel;
import java.awt.image.BufferedImage;

/** Compact quick-prediction panel placed on the left side of the main window. */
public class QuickPredictPanel extends Panel {

    private final DigitCanvas miniCanvas;
    private final Label resultDigitLabel;
    private final Label resultConfLabel;
    private final Label resultAltLabel;
    private final Label statusLabel;
    private final Choice modelChoice;
    private final Button predictButton;
    private final Button clearButton;

    public QuickPredictPanel() {
        setLayout(new BorderLayout(4, 4));
        setBackground(new Color(224, 232, 243));
        setPreferredSize(new Dimension(185, 0));

        // ── Title ──
        Label titleLabel = new Label("Quick Predict", Label.CENTER);
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 13));
        titleLabel.setBackground(new Color(31, 52, 81));
        titleLabel.setForeground(Color.WHITE);

        // ── Mini canvas ──
        miniCanvas = new DigitCanvas(150, 150);
        miniCanvas.setPreferredSize(new Dimension(155, 155));
        Panel canvasWrap = new Panel(new FlowLayout(FlowLayout.CENTER, 4, 4));
        canvasWrap.setBackground(new Color(224, 232, 243));
        canvasWrap.add(miniCanvas);

        // ── Buttons ──
        predictButton = new Button("Predict");
        predictButton.setBackground(new Color(39, 116, 172));
        predictButton.setForeground(Color.WHITE);
        predictButton.setFont(new Font("SansSerif", Font.BOLD, 11));

        clearButton = new Button("Clear");
        clearButton.setBackground(new Color(95, 108, 125));
        clearButton.setForeground(Color.WHITE);
        clearButton.setFont(new Font("SansSerif", Font.BOLD, 11));

        Panel buttonPanel = new Panel(new FlowLayout(FlowLayout.CENTER, 4, 2));
        buttonPanel.setBackground(new Color(224, 232, 243));
        buttonPanel.add(predictButton);
        buttonPanel.add(clearButton);

        // ── Results ──
        resultDigitLabel = new Label("Digit: –", Label.LEFT);
        resultDigitLabel.setFont(new Font("SansSerif", Font.BOLD, 18));
        resultDigitLabel.setForeground(new Color(31, 52, 81));

        resultConfLabel = new Label("Conf: –", Label.LEFT);
        resultConfLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));

        resultAltLabel = new Label("Alt: –", Label.LEFT);
        resultAltLabel.setFont(new Font("SansSerif", Font.PLAIN, 11));
        resultAltLabel.setForeground(new Color(100, 100, 130));

        statusLabel = new Label("Draw & predict", Label.CENTER);
        statusLabel.setFont(new Font("SansSerif", Font.ITALIC, 10));
        statusLabel.setForeground(new Color(100, 110, 130));

        Panel resultsPanel = new Panel(new GridLayout(4, 1, 0, 2));
        resultsPanel.setBackground(new Color(224, 232, 243));
        resultsPanel.add(resultDigitLabel);
        resultsPanel.add(resultConfLabel);
        resultsPanel.add(resultAltLabel);
        resultsPanel.add(statusLabel);

        // ── Model selector ──
        Label modelLabel = new Label("Model:", Label.LEFT);
        modelLabel.setFont(new Font("SansSerif", Font.PLAIN, 11));

        modelChoice = new Choice();
        modelChoice.setFont(new Font("SansSerif", Font.PLAIN, 11));
        for (String name : ModelFactory.getAvailableModelNames()) {
            modelChoice.add(name);
        }

        Panel modelPanel = new Panel(new GridLayout(2, 1));
        modelPanel.setBackground(new Color(224, 232, 243));
        modelPanel.add(modelLabel);
        modelPanel.add(modelChoice);

        // ── Assemble center ──
        Panel centerPanel = new Panel(new BorderLayout(4, 4));
        centerPanel.setBackground(new Color(224, 232, 243));
        centerPanel.add(canvasWrap, BorderLayout.NORTH);

        Panel controlsPanel = new Panel(new BorderLayout(4, 4));
        controlsPanel.setBackground(new Color(224, 232, 243));
        controlsPanel.add(buttonPanel, BorderLayout.NORTH);
        controlsPanel.add(resultsPanel, BorderLayout.CENTER);
        controlsPanel.add(modelPanel, BorderLayout.SOUTH);

        add(titleLabel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
        add(controlsPanel, BorderLayout.SOUTH);

        registerEvents();
    }

    private void registerEvents() {
        predictButton.addActionListener(event -> runPrediction());
        clearButton.addActionListener(event -> {
            miniCanvas.clear();
            resultDigitLabel.setText("Digit: –");
            resultConfLabel.setText("Conf: –");
            resultAltLabel.setText("Alt: –");
            statusLabel.setText("Draw & predict");
        });
    }

    private void runPrediction() {
        predictButton.setEnabled(false);
        statusLabel.setText("Predicting...");

        Thread worker = new Thread(() -> {
            try {
                BufferedImage processed = ImagePreprocessor.preprocess(
                        miniCanvas.getDrawingImage());
                String selectedModel = modelChoice.getSelectedItem();
                DigitModel model = ModelFactory.loadModelByName(selectedModel);
                PredictionResult result = model.predict(processed);

                EventQueue.invokeLater(() -> {
                    if (result.isValidDigit()) {
                        resultDigitLabel.setText("Digit: " + result.digit());
                        resultConfLabel.setText("Conf: "
                                + Math.round(result.confidence() * 100) + "%");
                        resultAltLabel.setText("Alt: " + result.alternativesSummary());
                        statusLabel.setText("Prediction complete");
                    } else {
                        resultDigitLabel.setText("Digit: ?");
                        resultConfLabel.setText("Conf: –");
                        resultAltLabel.setText("No valid digit detected");
                        statusLabel.setText("Try drawing a clearer digit");
                    }
                    predictButton.setEnabled(true);
                });
            } catch (Exception exception) {
                EventQueue.invokeLater(() -> {
                    statusLabel.setText("Error: " + exception.getMessage());
                    predictButton.setEnabled(true);
                });
            }
        }, "quick-predict-worker");
        worker.setDaemon(true);
        worker.start();
    }
}

