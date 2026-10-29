# DigitSense

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java: 17+](https://img.shields.io/badge/Java-17%2B-blue.svg)](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html)
[![Deeplearning4j](https://img.shields.io/badge/DL4J-1.0.0--M2.1-orange.svg)](https://deeplearning4j.konduit.ai/)
[![Database: SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)

DigitSense is an intelligent Java desktop application for handwritten digit recognition, binary number interpretation, and machine learning model benchmarking. It features an interactive drawing canvas, multi-digit image segmentation, real-time prediction with model selection, and a comprehensive model comparison suite supporting 6 distinct machine learning and deep learning architectures.

---

## Key Features

### 1. Interactive Drawing & Recognition
- **Full Drawing Canvas**: Smooth mouse-driven drawing area with multiple pen colours (Black, Blue, Red, Green, Purple, Orange).
- **Quick Predict Panel**: Dedicated mini-screen on the left sidebar for rapid sketching, immediate digit recognition, confidence scores, top alternatives, and real-time model switching.
- **Image Import**: Load external handwritten digit images in PNG, JPG, or JPEG format.
- **Multi-Digit Segmentation**: Automatically segments multi-digit inputs via vertical ink gap analysis.
- **Binary Conversion**: Interprets binary digits (`0` and `1`) with an interactive `8 4 2 1` place-value step-by-step breakdown into decimal.

### 2. Model Comparison & Benchmarking Suite
Compare 6 diverse machine learning and deep learning models evaluated on MNIST:
- **Logistic Regression**: Baseline linear classifier on raw 784-pixel inputs with softmax output.
- **k-Nearest Neighbors (KNN)**: Non-parametric classification using Euclidean distance voting ($k=3$).
- **Random Forest**: Ensemble of shallow neural networks with bagging and averaged probability distributions.
- **Multilayer Perceptron (MLP)**: Deep feedforward network ($784 \to 256 \to 128 \to 10$) with ReLU activations and Adam optimization.
- **LeNet-style CNN**: Classic convolutional architecture (Conv $5\times5 \to$ MaxPool $\to$ Conv $5\times5 \to$ MaxPool $\to$ Dense $\to$ Softmax).
- **Deeper CNN**: High-capacity convolutional neural network featuring Batch Normalization, Dropout ($0.25$ / $0.5$), and dual-stage convolutions.

### 3. Evaluation & Metrics Visualization
- **Performance Table**: Displays Accuracy, Macro Precision, Macro Recall, Macro F1-score, and Training/Evaluation Time.
- **10×10 Confusion Matrix Viewer**: Interactive heat-map dialog with colour-coded cells (green for correct predictions, red for classification errors) and numerical frequency counts.
- **Per-Class Metrics**: In-depth breakdown of Precision, Recall, and F1-score for each individual digit class ($0$ through $9$).

### 4. Persistence & Feedback Loop
- **SQLite Storage**: Automatically records predictions, timestamps, confidence scores, model versions, and inference latency in `data/digitsense.db`.
- **CSV Export**: Export all historical prediction logs to `data/exports/predictions.csv`.
- **User Correction & Fine-Tuning**: Consent-based feedback collection to store labeled canvas samples for continuous CNN fine-tuning.

---

## System Requirements

- **Java Development Kit (JDK)**: Version 17 or higher
- **Build Tool**: Apache Maven 3.9 or newer
- **Operating System**: Windows 10/11 (64-bit), Linux desktop (64-bit), or macOS
- **Display**: Graphical desktop environment (uses Java AWT)

---

## Installation & Running

### Clone the Repository
```bash
git clone https://github.com/Dasarath-Mali/DigitSense.git
cd DigitSense
```

### Build and Run

#### Linux / macOS
```bash
mvn clean compile
mvn exec:java
```

#### Windows PowerShell
```powershell
mvn clean compile
mvn exec:java
```

#### Running Tests
```bash
mvn test
```

---

## How to Use DigitSense

1. **Main Canvas**:
   - Draw single or multi-digit numbers using the main central canvas.
   - Choose pen colours and detection mode (`Auto detect`, `Normal digit mode`, or `Binary mode`).
   - Click **Predict** to recognize the digits and view prediction latency.
2. **Quick Predict (Left Panel)**:
   - Use the mini-canvas on the left sidebar for quick single-digit sketching.
   - Select your preferred model from the dropdown (e.g., Default CNN, Logistic Regression, MLP, LeNet, Deeper CNN).
   - Click **Predict** to view the classified digit, confidence percentage, and alternative rankings.
3. **Compare Models Tab**:
   - Click the **Compare Models** button on the top navigation bar.
   - Click **Run Benchmark** to train and evaluate all 6 models across MNIST.
   - Select any model row from the comparison table to activate:
     - **View Confusion Matrix**: Opens a 10×10 visual heatmap.
     - **Per-Class Metrics**: Displays individual digit precision, recall, and F1 scores.
4. **Correction & Feedback**:
   - Enter the correct digit in the bottom toolbar, tick **Consent**, and click **Correct** to save labeled data for fine-tuning.
5. **Exporting History**:
   - Click **Export CSV** to save complete prediction logs to `data/exports/predictions.csv`.

---

## Training & Fine-Tuning

### Train the Base CNN Model
```bash
mvn exec:java -Dexec.mainClass=com.digitsense.training.MnistCnnTrainer
```
This trains the default CNN on the MNIST dataset and outputs:
- `models/digitsense-cnn.zip`
- `models/digitsense-cnn-base.zip`

### Fine-Tune on User Canvas Samples
Once at least 5 consented samples per digit class ($0$–$9$) have been gathered:
```bash
mvn exec:java -Dexec.mainClass=com.digitsense.training.CanvasFineTuner
```

---

## Project Structure

```text
DigitSense/
├── src/main/java/com/digitsense/
│   ├── app/
│   │   └── Main.java                      # Application launcher
│   ├── binary/
│   │   └── BinaryNumberConverter.java     # Binary-to-decimal place-value calculation
│   ├── database/
│   │   └── DatabaseManager.java           # SQLite database operations & CSV export
│   ├── image/
│   │   ├── DigitSegmenter.java            # Multi-digit vertical ink-gap segmentation
│   │   └── ImagePreprocessor.java         # 28x28 cropping, scaling & center-of-mass
│   ├── model/
│   │   ├── DigitModel.java                # Unified model interface
│   │   ├── Dl4jDigitModel.java            # DL4J CNN implementation
│   │   ├── LogisticRegressionModel.java   # Logistic regression on raw pixels
│   │   ├── KnnDigitModel.java             # k-Nearest Neighbors classifier
│   │   ├── RandomForestModel.java         # Neural ensemble classifier
│   │   ├── MlpDigitModel.java             # Multilayer perceptron model
│   │   ├── LeNetDigitModel.java           # Classic LeNet-style CNN
│   │   ├── DeeperCnnDigitModel.java       # Deep CNN with BatchNorm & Dropout
│   │   ├── ModelBenchmark.java            # Automated multi-model evaluation suite
│   │   ├── ModelMetrics.java              # Accuracy, F1, precision, recall & matrix container
│   │   ├── ModelPredictionHelper.java     # Shared probability & top-N extractor
│   │   ├── PredictionResult.java          # Prediction result record
│   │   └── UntrainedDigitModel.java       # Standby model
│   ├── training/
│   │   ├── MnistCnnTrainer.java           # Base MNIST training script
│   │   └── CanvasFineTuner.java           # Feedback fine-tuning script
│   └── ui/
│       ├── DigitCanvas.java               # Smooth mouse drawing canvas
│       ├── DigitSenseFrame.java           # Main AWT window with tabbed layout
│       ├── QuickPredictPanel.java         # Left-side mini prediction panel
│       ├── CompareModelsPanel.java        # Model comparison dashboard & table
│       ├── ConfusionMatrixDialog.java     # Visual 10x10 confusion matrix heatmap
│       └── PerClassMetricsDialog.java     # Per-digit precision/recall/F1 dialog
├── src/test/java/                         # Unit and integration tests
├── models/                                # Serialized DL4J models (.zip)
├── docs/                                  # Architecture and design specifications
├── LICENSE                                # MIT License
├── pom.xml                                # Maven build and dependency configuration
└── README.md                              # Project documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## Author

Created by [Dasarath Mali](https://github.com/Dasarath-Mali).
