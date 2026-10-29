package com.digitsense.ui;

import com.digitsense.model.ModelMetrics;

import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Color;
import java.awt.Dialog;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Label;
import java.awt.Panel;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/** Dialog showing per-class precision, recall, and F1 for a selected model. */
public class PerClassMetricsDialog extends Dialog {

    public PerClassMetricsDialog(Frame parent, ModelMetrics metrics) {
        super(parent, "Per-Class Metrics - " + metrics.getModelName(), true);

        setLayout(new BorderLayout(8, 8));
        setBackground(new Color(235, 240, 248));

        Label titleLabel = new Label("Per-Class Metrics: " + metrics.getModelName(), Label.CENTER);
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 14));
        titleLabel.setBackground(new Color(31, 52, 81));
        titleLabel.setForeground(Color.WHITE);

        // Header
        Label header = new Label(
                "  Digit    Precision     Recall        F1");
        header.setFont(new Font("Monospaced", Font.BOLD, 13));
        header.setBackground(new Color(200, 212, 228));

        // Data rows
        java.awt.List dataList = new java.awt.List(10, false);
        dataList.setFont(new Font("Monospaced", Font.PLAIN, 13));
        dataList.setBackground(Color.WHITE);

        double[] prec = metrics.getPrecisionPerClass();
        double[] rec = metrics.getRecallPerClass();
        double[] f1 = metrics.getF1PerClass();

        for (int digit = 0; digit < 10; digit++) {
            dataList.add(String.format("    %d      %6.2f%%       %6.2f%%      %6.2f%%",
                    digit, prec[digit] * 100, rec[digit] * 100, f1[digit] * 100));
        }
        dataList.add("  ─────────────────────────────────────────");
        dataList.add(String.format("  Macro    %6.2f%%       %6.2f%%      %6.2f%%",
                metrics.getMacroPrecision() * 100,
                metrics.getMacroRecall() * 100,
                metrics.getMacroF1() * 100));
        dataList.add(String.format("  Overall accuracy: %6.2f%%", metrics.getAccuracy() * 100));

        Button closeButton = new Button("Close");
        closeButton.setBackground(new Color(95, 108, 125));
        closeButton.setForeground(Color.WHITE);
        closeButton.addActionListener(event -> dispose());

        Panel contentPanel = new Panel(new BorderLayout(4, 4));
        contentPanel.add(header, BorderLayout.NORTH);
        contentPanel.add(dataList, BorderLayout.CENTER);

        Panel bottomPanel = new Panel(new FlowLayout(FlowLayout.CENTER));
        bottomPanel.add(closeButton);

        add(titleLabel, BorderLayout.NORTH);
        add(contentPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);

        setSize(420, 380);
        setLocationRelativeTo(parent);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent event) {
                dispose();
            }
        });
    }
}

