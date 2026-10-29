package com.digitsense.ui;

import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dialog;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.FlowLayout;
import java.awt.Label;
import java.awt.Panel;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/** Dialog that displays a 10x10 confusion matrix as a painted heat-map grid. */
public class ConfusionMatrixDialog extends Dialog {

    private final int[][] matrix;

    public ConfusionMatrixDialog(Frame parent, String modelName, int[][] confusionMatrix) {
        super(parent, "Confusion Matrix - " + modelName, true);
        this.matrix = confusionMatrix;

        setLayout(new BorderLayout(8, 8));
        setBackground(new Color(235, 240, 248));

        Label titleLabel = new Label("Confusion Matrix: " + modelName, Label.CENTER);
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 14));
        titleLabel.setBackground(new Color(31, 52, 81));
        titleLabel.setForeground(Color.WHITE);

        Label legendLabel = new Label(
                "Rows = Actual, Columns = Predicted  |  Green = Correct, Red = Error",
                Label.CENTER);
        legendLabel.setFont(new Font("SansSerif", Font.PLAIN, 11));

        Canvas matrixCanvas = new Canvas() {
            @Override
            public void paint(Graphics g) {
                drawMatrix(g, getWidth(), getHeight());
            }

            @Override
            public Dimension getPreferredSize() {
                return new Dimension(460, 460);
            }
        };
        matrixCanvas.setBackground(Color.WHITE);

        Button closeButton = new Button("Close");
        closeButton.setBackground(new Color(95, 108, 125));
        closeButton.setForeground(Color.WHITE);
        closeButton.addActionListener(event -> dispose());

        Panel bottomPanel = new Panel(new FlowLayout(FlowLayout.CENTER));
        bottomPanel.add(legendLabel);
        bottomPanel.add(closeButton);

        add(titleLabel, BorderLayout.NORTH);
        add(matrixCanvas, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);

        setSize(520, 560);
        setLocationRelativeTo(parent);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent event) {
                dispose();
            }
        });
    }

    private void drawMatrix(Graphics g, int width, int height) {
        int margin = 40;
        int cellWidth = (width - margin * 2) / 11;
        int cellHeight = (height - margin * 2) / 11;

        // Find max value for scaling colours
        int maxValue = 1;
        for (int[] row : matrix) {
            for (int value : row) {
                maxValue = Math.max(maxValue, value);
            }
        }

        Font headerFont = new Font("SansSerif", Font.BOLD, 11);
        Font cellFont = new Font("SansSerif", Font.PLAIN, 10);

        // Draw column headers
        g.setFont(headerFont);
        g.setColor(new Color(31, 52, 81));
        g.drawString("Pred→", margin, margin - 5);
        for (int col = 0; col < 10; col++) {
            int x = margin + (col + 1) * cellWidth;
            g.drawString(String.valueOf(col), x + cellWidth / 2 - 3, margin - 5);
        }

        // Draw rows
        for (int row = 0; row < 10; row++) {
            int y = margin + (row + 1) * cellHeight;

            // Row header
            g.setFont(headerFont);
            g.setColor(new Color(31, 52, 81));
            g.drawString(String.valueOf(row), margin + cellWidth / 2 - 3, y + cellHeight / 2 + 4);

            // Cells
            g.setFont(cellFont);
            for (int col = 0; col < 10; col++) {
                int x = margin + (col + 1) * cellWidth;
                int value = matrix[row][col];
                float intensity = (float) value / maxValue;

                // Colour: green for diagonal, red for off-diagonal
                Color cellColor;
                if (row == col) {
                    int green = 180 + (int) (75 * intensity);
                    cellColor = new Color(220 - (int) (120 * intensity),
                            Math.min(255, green), 220 - (int) (120 * intensity));
                } else {
                    cellColor = new Color(
                            Math.min(255, 220 + (int) (35 * intensity)),
                            220 - (int) (160 * intensity),
                            220 - (int) (160 * intensity));
                }

                g.setColor(cellColor);
                g.fillRect(x, y, cellWidth - 1, cellHeight - 1);

                // Border
                g.setColor(new Color(180, 190, 200));
                g.drawRect(x, y, cellWidth - 1, cellHeight - 1);

                // Value text
                if (value > 0) {
                    g.setColor(intensity > 0.6 ? Color.WHITE : Color.BLACK);
                    String text = String.valueOf(value);
                    int textWidth = g.getFontMetrics().stringWidth(text);
                    g.drawString(text, x + (cellWidth - textWidth) / 2,
                            y + cellHeight / 2 + 4);
                }
            }
        }

        // "Actual" label on the left
        g.setFont(headerFont);
        g.setColor(new Color(31, 52, 81));
        g.drawString("Act↓", margin, margin + cellHeight / 2 + 4);
    }
}

