package com.digitsense.ui;

import com.digitsense.binary.BinaryNumberConverter;
import com.digitsense.database.DatabaseManager;
import com.digitsense.image.DigitSegmenter;
import com.digitsense.image.ImagePreprocessor;
import com.digitsense.model.DigitModel;
import com.digitsense.model.ModelFactory;
import com.digitsense.model.PredictionResult;

import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.CardLayout;
import java.awt.Checkbox;
import java.awt.Color;
import java.awt.Choice;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.FileDialog;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Label;
import java.awt.Panel;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import javax.imageio.ImageIO;

/** Main AWT window for DigitSense. */
public class DigitSenseFrame extends Frame implements ActionListener {

    private final DigitCanvas digitCanvas;
    private final Label predictionLabel;
    private final Label alternativesLabel;
    private final Label conversionLabel;
    private final Button predictButton;
    private final Button openImageButton;
    private final Button clearButton;
    private final Button saveButton;
    private final Button exportButton;
    private final Button correctButton;
    private final Choice penColorChoice;
    private final Choice numberModeChoice;
    private final TextField correctionField;
    private final Checkbox consentCheckbox;
    private final java.awt.List historyList;
    private Panel historyPanel;
    private int historyPanelWidth = 210;
    private DatabaseManager databaseManager;
    private final DigitModel digitModel;
    private long lastPredictionId = -1;
    private BufferedImage lastProcessedImage;
    private int lastPredictionCount;

    // ── New panels ──
    private final QuickPredictPanel quickPredictPanel;
    private final CompareModelsPanel compareModelsPanel;
    private final Button viewCanvasButton;
    private final Button viewCompareButton;
    private final CardLayout centerCardLayout;
    private final Panel centerCardPanel;

    public DigitSenseFrame() {
        super("DigitSense - Handwritten Digit Recognizer");

        digitCanvas = new DigitCanvas(500, 400);
        predictionLabel = new Label("Draw a digit and click Predict", Label.CENTER);
        alternativesLabel = new Label("", Label.CENTER);
        conversionLabel = new Label("", Label.CENTER);
        predictButton = new Button("Predict");
        openImageButton = new Button("Open Image");
        clearButton = new Button("Clear");
        saveButton = new Button("Save");
        exportButton = new Button("Export CSV");
        correctButton = new Button("Correct");
        penColorChoice = new Choice();
        penColorChoice.add("Black");
        penColorChoice.add("Blue");
        penColorChoice.add("Red");
        penColorChoice.add("Green");
        penColorChoice.add("Purple");
        penColorChoice.add("Orange");
        numberModeChoice = new Choice();
        numberModeChoice.add("Auto detect");
        numberModeChoice.add("Normal digit mode");
        numberModeChoice.add("Binary mode (0 and 1)");
        correctionField = new TextField(2);
        consentCheckbox = new Checkbox("Consent");
        historyList = new java.awt.List(8, false);
        digitModel = ModelFactory.loadAvailableModel();

        // New components
        quickPredictPanel = new QuickPredictPanel();
        compareModelsPanel = new CompareModelsPanel(this);
        viewCanvasButton = new Button("Drawing Canvas");
        viewCompareButton = new Button("Compare Models");
        centerCardLayout = new CardLayout();
        centerCardPanel = new Panel(centerCardLayout);

        initializeDatabase();
        buildWindow();
        registerEvents();
        refreshHistory();
    }

    private void initializeDatabase() {
        try {
            databaseManager = new DatabaseManager();
        } catch (SQLException exception) {
            databaseManager = null;
            System.err.println("Database could not be initialized: " + exception.getMessage());
        }
    }

    private void buildWindow() {
        setLayout(new BorderLayout(10, 10));
        setBackground(new Color(235, 240, 248));

        // ── Top panel with prediction display ──
        Panel topPanel = new Panel(new GridLayout(3, 1));
        topPanel.setBackground(new Color(31, 52, 81));
        predictionLabel.setForeground(Color.WHITE);
        predictionLabel.setFont(new Font("SansSerif", Font.BOLD, 18));
        alternativesLabel.setForeground(new Color(210, 225, 245));
        conversionLabel.setForeground(new Color(255, 220, 135));
        topPanel.add(predictionLabel);
        topPanel.add(alternativesLabel);
        topPanel.add(conversionLabel);

        // ── Tab buttons ──
        Panel tabPanel = new Panel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        tabPanel.setBackground(new Color(31, 52, 81));
        applyTabStyle(viewCanvasButton, true);
        applyTabStyle(viewCompareButton, false);
        tabPanel.add(viewCanvasButton);
        tabPanel.add(viewCompareButton);

        Panel topBlock = new Panel(new BorderLayout());
        topBlock.add(topPanel, BorderLayout.CENTER);
        topBlock.add(tabPanel, BorderLayout.SOUTH);

        // ── Bottom panel with controls ──
        Panel bottomPanel = new Panel(new FlowLayout(FlowLayout.CENTER));
        bottomPanel.setBackground(new Color(218, 228, 241));
        bottomPanel.add(new Label("Pen colour:"));
        bottomPanel.add(penColorChoice);
        bottomPanel.add(new Label("Mode:"));
        bottomPanel.add(numberModeChoice);
        bottomPanel.add(openImageButton);
        bottomPanel.add(predictButton);
        bottomPanel.add(clearButton);
        bottomPanel.add(saveButton);
        bottomPanel.add(exportButton);
        bottomPanel.add(new Label("Correct digit:"));
        bottomPanel.add(correctionField);
        bottomPanel.add(consentCheckbox);
        bottomPanel.add(correctButton);

        // ── History sidebar (right) ──
        historyPanel = new Panel(new BorderLayout());
        historyPanel.setBackground(new Color(224, 232, 243));
        historyPanel.setPreferredSize(new Dimension(historyPanelWidth, 0));
        historyPanel.add(new Label("Recent predictions", Label.CENTER), BorderLayout.NORTH);
        historyPanel.add(historyList, BorderLayout.CENTER);
        addResizableDivider(historyPanel);

        // ── Card panel: Drawing canvas vs Compare models ──
        Panel canvasWorkspace = new Panel(new BorderLayout(8, 0));
        canvasWorkspace.setBackground(Color.WHITE);
        canvasWorkspace.add(digitCanvas, BorderLayout.CENTER);
        canvasWorkspace.add(historyPanel, BorderLayout.EAST);

        centerCardPanel.add(canvasWorkspace, "canvas");
        centerCardPanel.add(compareModelsPanel, "compare");

        // ── Main workspace: QuickPredict on left + cards in center ──
        Panel workspacePanel = new Panel(new BorderLayout(6, 0));
        workspacePanel.setBackground(new Color(235, 240, 248));
        workspacePanel.add(quickPredictPanel, BorderLayout.WEST);
        workspacePanel.add(centerCardPanel, BorderLayout.CENTER);

        add(topBlock, BorderLayout.NORTH);
        add(workspacePanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);

        setSize(950, 650);
        setLocationRelativeTo(null);
        setMinimumSize(new Dimension(900, 600));
        applyButtonStyle(predictButton, new Color(39, 116, 172));
        applyButtonStyle(clearButton, new Color(95, 108, 125));
        applyButtonStyle(saveButton, new Color(48, 137, 94));
        applyButtonStyle(exportButton, new Color(104, 83, 153));
        applyButtonStyle(correctButton, new Color(190, 116, 36));
    }

    private void applyButtonStyle(Button button, Color background) {
        button.setBackground(background);
        button.setForeground(Color.WHITE);
    }

    private void applyTabStyle(Button button, boolean active) {
        if (active) {
            button.setBackground(new Color(39, 116, 172));
            button.setForeground(Color.WHITE);
        } else {
            button.setBackground(new Color(60, 80, 110));
            button.setForeground(new Color(180, 195, 215));
        }
        button.setFont(new Font("SansSerif", Font.BOLD, 12));
    }

    /** Adds a draggable divider so the history sidebar can be resized at runtime. */
    private void addResizableDivider(Panel panel) {
        Panel divider = new Panel();
        divider.setBackground(new Color(166, 181, 201));
        divider.setPreferredSize(new Dimension(10, 0));
        divider.setCursor(Cursor.getPredefinedCursor(Cursor.W_RESIZE_CURSOR));
        panel.add(divider, BorderLayout.WEST);
        final int[] dragStartX = {0};
        final int[] dragStartWidth = {historyPanelWidth};

        divider.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent event) {
                dragStartX[0] = event.getXOnScreen();
                dragStartWidth[0] = historyPanelWidth;
            }
        });
        divider.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent event) {
                int change = dragStartX[0] - event.getXOnScreen();
                historyPanelWidth = Math.max(150,
                        Math.min(420, dragStartWidth[0] + change));
                panel.setPreferredSize(new Dimension(historyPanelWidth, 0));
                validate();
            }
        });
    }

    private void registerEvents() {
        predictButton.addActionListener(this);
        openImageButton.addActionListener(this);
        clearButton.addActionListener(this);
        saveButton.addActionListener(this);
        exportButton.addActionListener(this);
        correctButton.addActionListener(this);
        penColorChoice.addItemListener(event -> digitCanvas.setPenColor(selectedPenColor()));

        // Tab switching
        viewCanvasButton.addActionListener(event -> {
            centerCardLayout.show(centerCardPanel, "canvas");
            applyTabStyle(viewCanvasButton, true);
            applyTabStyle(viewCompareButton, false);
        });
        viewCompareButton.addActionListener(event -> {
            centerCardLayout.show(centerCardPanel, "compare");
            applyTabStyle(viewCanvasButton, false);
            applyTabStyle(viewCompareButton, true);
        });

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent event) {
                dispose();
                System.exit(0);
            }
        });
    }

    private Color selectedPenColor() {
        return switch (penColorChoice.getSelectedItem()) {
            case "Blue" -> new Color(25, 88, 166);
            case "Red" -> new Color(196, 49, 49);
            case "Green" -> new Color(30, 130, 76);
            case "Purple" -> new Color(116, 70, 158);
            case "Orange" -> new Color(220, 116, 24);
            default -> Color.BLACK;
        };
    }

    @Override
    public void actionPerformed(ActionEvent event) {
        if (event.getSource() == clearButton) {
            digitCanvas.clear();
            predictionLabel.setText("Draw a digit and click Predict");
            alternativesLabel.setText("");
            conversionLabel.setText("");
            lastPredictionCount = 0;
        } else if (event.getSource() == predictButton) {
            startPrediction();
        } else if (event.getSource() == openImageButton) {
            openImage();
        } else if (event.getSource() == saveButton) {
            saveImage(digitCanvas.getDrawingImage(), new File("data/canvas-samples/drawing.png"));
            predictionLabel.setText("Drawing saved to data/canvas-samples/drawing.png");
        } else if (event.getSource() == exportButton) {
            exportHistory();
        } else if (event.getSource() == correctButton) {
            saveCorrection();
        }
    }

    private void exportHistory() {
        if (databaseManager == null) {
            predictionLabel.setText("Database is unavailable");
            return;
        }
        try {
            databaseManager.exportPredictions(new File("data/exports/predictions.csv"));
            predictionLabel.setText("History exported to data/exports/predictions.csv");
        } catch (SQLException | IOException exception) {
            predictionLabel.setText("History export failed");
            System.err.println("History export failed: " + exception.getMessage());
        }
    }

    private void startPrediction() {
        predictButton.setEnabled(false);
        predictionLabel.setText("Processing prediction...");
        alternativesLabel.setText("");
        conversionLabel.setText("");
        String selectedMode = numberModeChoice.getSelectedItem();

        Thread predictionThread = new Thread(() -> {
            PredictionOutcome outcome = computePrediction(selectedMode);
            EventQueue.invokeLater(() -> applyPredictionOutcome(outcome));
        }, "digitsense-prediction-worker");
        predictionThread.start();
    }

    private void openImage() {
        FileDialog dialog = new FileDialog(this, "Open handwritten digit image", FileDialog.LOAD);
        dialog.setFile("*.png;*.jpg;*.jpeg");
        dialog.setVisible(true);
        if (dialog.getFile() == null) {
            return;
        }

        File selectedFile = new File(dialog.getDirectory(), dialog.getFile());
        try {
            BufferedImage image = ImageIO.read(selectedFile);
            if (image == null) {
                predictionLabel.setText("Unsupported image format");
                return;
            }
            digitCanvas.loadImage(image);
            predictionLabel.setText("Image loaded: " + selectedFile.getName());
            alternativesLabel.setText("Click Predict to recognize the image");
            conversionLabel.setText("");
            lastPredictionCount = 0;
            lastProcessedImage = null;
        } catch (IOException exception) {
            predictionLabel.setText("Could not open image");
            System.err.println("Image loading failed: " + exception.getMessage());
        }
    }

    private PredictionOutcome computePrediction(String selectedMode) {
        List<BufferedImage> segments = DigitSegmenter.segment(digitCanvas.getDrawingImage());
        if (segments.isEmpty()) {
            return PredictionOutcome.error("Draw at least one digit first");
        }

        StringBuilder predictedText = new StringBuilder("Prediction: ");
        StringBuilder alternatives = new StringBuilder("Other possibilities: ");
        StringBuilder binaryBits = new StringBuilder();
        boolean binaryCandidate = true;
        BufferedImage processedSingleImage = null;
        long predictionId = -1;
        double totalLatencyMs = 0.0;

        try {
            for (int index = 0; index < segments.size(); index++) {
                BufferedImage processedImage = ImagePreprocessor.preprocess(segments.get(index));
                long predictionStart = System.nanoTime();
                PredictionResult result = digitModel.predict(processedImage);
                double latencyMs = (System.nanoTime() - predictionStart) / 1_000_000.0;
                totalLatencyMs += latencyMs;
                if (segments.size() == 1) {
                    processedSingleImage = processedImage;
                    writeImage(processedImage, new File("data/preprocessed/last-input.png"));
                } else {
                    writeImage(processedImage,
                            new File("data/preprocessed/digit-" + (index + 1) + ".png"));
                }

                if (result.isValidDigit()) {
                    predictedText.append(result.digit());
                    if (result.digit() == 0 || result.digit() == 1) {
                        binaryBits.append(result.digit());
                    } else {
                        binaryCandidate = false;
                    }
                    if (index > 0) {
                        alternatives.append(" | ");
                    }
                    alternatives.append("digit ").append(index + 1).append(": ")
                            .append(result.alternativesSummary());
                } else {
                    predictedText.append("?");
                    binaryCandidate = false;
                }

                if (databaseManager != null) {
                    predictionId = databaseManager.savePrediction(processedImage,
                            result.digit(), result.confidence(), result.modelVersion(), latencyMs);
                }
            }
            predictedText.append(" (" + String.format("%.1f", totalLatencyMs) + " ms)");
            String conversionText = "";
            boolean binaryMode = selectedMode.equals("Binary mode (0 and 1)")
                    || selectedMode.equals("Auto detect") && binaryCandidate;
            if (binaryMode) {
                if (!binaryCandidate) {
                    conversionText = "Binary mode accepts only 0 and 1";
                } else {
                    BinaryNumberConverter.Conversion conversion =
                            BinaryNumberConverter.convert(binaryBits.toString());
                    conversionText = conversion.displayText();
                    predictedText = new StringBuilder("Decimal prediction: ")
                            .append(conversion.decimal())
                            .append(" (")
                            .append(String.format("%.1f", totalLatencyMs))
                            .append(" ms)");
                }
            } else if (selectedMode.equals("Auto detect")) {
                conversionText = "Detected type: Normal decimal number";
            }
            return new PredictionOutcome(predictedText.toString(), alternatives.toString(),
                    conversionText, predictionId, segments.size(), processedSingleImage, null);
        } catch (SQLException | IOException exception) {
            return PredictionOutcome.error("Prediction could not be saved");
        }
    }

    private void applyPredictionOutcome(PredictionOutcome outcome) {
        predictButton.setEnabled(true);
        predictionLabel.setText(outcome.predictionText());
        alternativesLabel.setText(outcome.alternativesText());
        conversionLabel.setText(outcome.conversionText());
        if (outcome.errorMessage() != null) {
            return;
        }

        lastPredictionId = outcome.predictionId();
        lastPredictionCount = outcome.predictionCount();
        lastProcessedImage = outcome.processedImage();
        refreshHistory();
    }

    private void refreshHistory() {
        if (databaseManager == null) {
            return;
        }
        try {
            historyList.removeAll();
            for (String summary : databaseManager.getRecentPredictionSummaries(8)) {
                historyList.add(summary);
            }
        } catch (SQLException exception) {
            historyList.removeAll();
            historyList.add("History unavailable");
        }
    }

    private void saveCorrection() {
        if (lastPredictionCount > 1) {
            predictionLabel.setText("Correction currently supports one digit at a time");
            return;
        }
        if (databaseManager == null || lastPredictionId < 0) {
            predictionLabel.setText("Click Predict before saving a correction");
            return;
        }

        try {
            int correctDigit = Integer.parseInt(correctionField.getText().trim());
            if (correctDigit < 0 || correctDigit > 9) {
                predictionLabel.setText("Correct digit must be between 0 and 9");
                return;
            }
            if (!consentCheckbox.getState()) {
                predictionLabel.setText("Select Consent before saving feedback");
                return;
            }

            databaseManager.saveFeedback(lastPredictionId, correctDigit, true);
            saveTrainingSample(processedImageForFeedback(), correctDigit);
            predictionLabel.setText("Correction saved for record #" + lastPredictionId);
            correctionField.setText("");
        } catch (NumberFormatException exception) {
            predictionLabel.setText("Enter one digit from 0 to 9");
        } catch (SQLException exception) {
            predictionLabel.setText("Correction could not be saved");
            System.err.println("Feedback save failed: " + exception.getMessage());
        } catch (IOException exception) {
            predictionLabel.setText("Feedback saved, but sample could not be stored");
            System.err.println("Training sample save failed: " + exception.getMessage());
        }
    }

    private BufferedImage processedImageForFeedback() {
        if (lastProcessedImage != null) {
            return lastProcessedImage;
        }
        return ImagePreprocessor.preprocess(digitCanvas.getDrawingImage());
    }

    private void saveTrainingSample(BufferedImage image, int digit) throws IOException {
        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS"));
        File file = new File("data/canvas-training/" + digit + "/sample-"
                + timestamp + ".png");
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }
        ImageIO.write(image, "png", file);
    }

    private void saveImage(BufferedImage image, File file) {
        try {
            writeImage(image, file);
        } catch (IOException exception) {
            predictionLabel.setText("Could not save image: " + exception.getMessage());
        }
    }

    private void writeImage(BufferedImage image, File file) throws IOException {
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }
        ImageIO.write(image, "png", file);
    }

    private record PredictionOutcome(String predictionText, String alternativesText,
                                     String conversionText,
                                     long predictionId, int predictionCount,
                                     BufferedImage processedImage, String errorMessage) {
        private static PredictionOutcome error(String message) {
            return new PredictionOutcome(message, "", "", -1, 0, null, message);
        }
    }
}
