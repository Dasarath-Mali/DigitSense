package com.digitsense.database;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

/** Creates the local SQLite database and stores DigitSense prediction records. */
public class DatabaseManager {

    private static final String DATABASE_URL = "jdbc:sqlite:data/digitsense.db";

    public DatabaseManager() throws SQLException {
        File dataDir = new File("data");
        if (!dataDir.exists()) {
            dataDir.mkdirs();
        }
        createTables();
    }

    public long savePrediction(BufferedImage image, int predictedDigit,
                               double confidence, String modelVersion)
            throws SQLException, IOException {
        return savePrediction(image, predictedDigit, confidence, modelVersion, null);
    }

    public long savePrediction(BufferedImage image, int predictedDigit,
                               double confidence, String modelVersion,
                               Double latencyMs)
            throws SQLException, IOException {
        String sql = "INSERT INTO predictions "
                + "(image_blob, predicted_digit, confidence, model_version, latency_ms) "
                + "VALUES (?, ?, ?, ?, ?)";

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(sql,
                     Statement.RETURN_GENERATED_KEYS)) {

            statement.setBytes(1, toPngBytes(image));
            statement.setInt(2, predictedDigit);
            statement.setDouble(3, confidence);
            statement.setString(4, modelVersion);
            if (latencyMs == null) {
                statement.setNull(5, java.sql.Types.REAL);
            } else {
                statement.setDouble(5, latencyMs);
            }
            statement.executeUpdate();

            try (var keys = statement.getGeneratedKeys()) {
                if (keys.next()) {
                    return keys.getLong(1);
                }
            }
        }

        throw new SQLException("The prediction record ID was not generated");
    }

    public void saveFeedback(long predictionId, int correctDigit,
                             boolean consentGiven) throws SQLException {
        String sql = "INSERT INTO feedback "
                + "(prediction_id, correct_digit, consent_flag) VALUES (?, ?, ?)";

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(sql)) {
            statement.setLong(1, predictionId);
            statement.setInt(2, correctDigit);
            statement.setBoolean(3, consentGiven);
            statement.executeUpdate();
        }
    }

    public List<String> getRecentPredictionSummaries(int limit) throws SQLException {
        String sql = "SELECT id, timestamp, predicted_digit, confidence "
                + "FROM predictions ORDER BY id DESC LIMIT ?";
        List<String> summaries = new ArrayList<>();

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(sql)) {
            statement.setInt(1, limit);
            try (ResultSet results = statement.executeQuery()) {
                while (results.next()) {
                    summaries.add("#" + results.getLong("id") + "  "
                            + results.getString("timestamp") + "  digit="
                            + results.getInt("predicted_digit") + "  "
                            + Math.round(results.getDouble("confidence") * 100) + "%");
                }
            }
        }
        return summaries;
    }

    public void saveModelRun(String modelName, double accuracy, double macroF1,
                             Double ece, Double latencyMs, String trainedOn,
                             long seed) throws SQLException {
        String sql = "INSERT INTO model_runs "
                + "(model_name, accuracy, macro_f1, ece, latency_ms, trained_on, seed) "
                + "VALUES (?, ?, ?, ?, ?, ?, ?)";

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(sql)) {
            statement.setString(1, modelName);
            statement.setDouble(2, accuracy);
            statement.setDouble(3, macroF1);
            if (ece == null) {
                statement.setNull(4, java.sql.Types.REAL);
            } else {
                statement.setDouble(4, ece);
            }
            if (latencyMs == null) {
                statement.setNull(5, java.sql.Types.REAL);
            } else {
                statement.setDouble(5, latencyMs);
            }
            statement.setString(6, trainedOn);
            statement.setLong(7, seed);
            statement.executeUpdate();
        }
    }

    public void exportPredictions(File file) throws SQLException, IOException {
        String sql = "SELECT id, timestamp, predicted_digit, confidence, "
                + "model_version, latency_ms FROM predictions ORDER BY id";
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }

        try (Connection connection = openConnection();
             PreparedStatement statement = connection.prepareStatement(sql);
             ResultSet results = statement.executeQuery();
             BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {

            writer.write("id,timestamp,predicted_digit,confidence,model_version,latency_ms");
            writer.newLine();
            while (results.next()) {
                writer.write(results.getLong("id") + ","
                        + results.getString("timestamp") + ","
                        + results.getInt("predicted_digit") + ","
                        + results.getDouble("confidence") + ","
                        + results.getString("model_version") + ","
                        + results.getObject("latency_ms"));
                writer.newLine();
            }
        }
    }

    private void createTables() throws SQLException {
        String predictions = "CREATE TABLE IF NOT EXISTS predictions ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
                + "image_blob BLOB NOT NULL, "
                + "predicted_digit INTEGER NOT NULL, "
                + "confidence REAL NOT NULL, "
                + "model_version TEXT NOT NULL)";

        String feedback = "CREATE TABLE IF NOT EXISTS feedback ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "prediction_id INTEGER NOT NULL, "
                + "correct_digit INTEGER NOT NULL, "
                + "consent_flag INTEGER NOT NULL DEFAULT 0, "
                + "FOREIGN KEY (prediction_id) REFERENCES predictions(id))";

        String modelRuns = "CREATE TABLE IF NOT EXISTS model_runs ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "model_name TEXT NOT NULL, "
                + "accuracy REAL, "
                + "macro_f1 REAL, "
                + "ece REAL, "
                + "latency_ms REAL, "
                + "trained_on TEXT, "
                + "seed INTEGER)";

        try (Connection connection = openConnection();
             Statement statement = connection.createStatement()) {
            statement.executeUpdate(predictions);
            try {
                statement.executeUpdate("ALTER TABLE predictions ADD COLUMN latency_ms REAL");
            } catch (SQLException ignored) {
                // The column already exists in an existing database.
            }
            statement.executeUpdate(feedback);
            statement.executeUpdate(modelRuns);
        }
    }

    private Connection openConnection() throws SQLException {
        return DriverManager.getConnection(DATABASE_URL);
    }

    private byte[] toPngBytes(BufferedImage image) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(image, "png", output);
        return output.toByteArray();
    }
}

