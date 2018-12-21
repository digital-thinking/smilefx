package de.ixeption.smilefx.controller;

import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.training.GridSearch;
import de.ixeption.smilefx.training.SmileModelTrainer;
import de.ixeption.smilefx.training.TrainedBinarySmileModel;
import de.ixeption.smilefx.training.TrainingDataSet;
import de.ixeption.smilefx.util.ModelManager;
import de.ixeption.smilefx.util.PersistenceUtils;
import de.ixeption.smilefx.util.RocCurve;
import de.ixeption.smilefx.util.TablesawConverter;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.Pane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.apache.commons.lang3.tuple.Triple;
import smile.feature.Scaler;
import smile.math.SparseArray;
import smile.math.kernel.*;
import smile.projection.*;
import smile.validation.*;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvWriteOptions;
import tech.tablesaw.io.csv.CsvWriter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;

import static java.util.stream.Collectors.joining;


public abstract class AbstractController<T, R> {

    public TextField samplingFactor;
    public Label fetchResult;
    public Accordion accordion;
    public TextField numThreads;
    public CheckBox SVM_Gaussian;
    public CheckBox SVM_Linear;
    public CheckBox SVM_Laplacian;
    public CheckBox AdaBoost;
    public CheckBox RandomForest;
    public CheckBox GradientBoostedTree;
    public ProgressBar trainingProgress;
    public ProgressBar fetchProgress;
    public ProgressBar sampleAndScaleProgress;
    public TableView<GridSearch.GridSearchResult> resultRecordTable;
    public TableColumn<GridSearch.GridSearchResult, String> modelColumn, paramsColumn;
    public TableColumn<GridSearch.GridSearchResult, Double> f1scoreColumn, mccScoreColumn, accuracyColumn, precisionColumn, recallColumn;
    public BarChart<String, Number> classesBarChart;
    public LineChart<Number, Number> rocCurve;
    public TextField limitData;
    public TableView<Triple<String, String, String>> confusionMatrix;
    public TableColumn<Triple<String, String, String>, String> placeholderColumn;
    public TableColumn<Triple<String, String, String>, String> negativeColumn;
    public TableColumn<Triple<String, String, String>, String> positiveColumn;
    public Button exportButton;
    public CheckBox scaleCheckbox;
    public CheckBox ppcaCheckBox;
    public CheckBox ghaCheckBox;
    public CheckBox gausskpca;
    public CheckBox htangentkpca;
    public CheckBox linearkpca;
    public TextField pcaDimensions;
    public Button gridSearchButton;
    public Button loadModelButton;
    public Label currentModelLabel;
    public Button verifyButton;
    public Label noDataLabel;
    public LineChart<Number, Number> resultROC;
    public TableView<Triple<String, String, String>> confusionMatrixVerify;
    public TableColumn<Triple<String, String, String>, String> placeholderColumnVerify;
    public TableColumn<Triple<String, String, String>, String> noConversionColumnVerify;
    public TableColumn<Triple<String, String, String>, String> conversionColumnVerify;
    public Button fetchButton;
    public Button sampleAndScaleButton;
    public Label modelInfoLabel;
    public Button updateThreshold;
    public Pane dataInput;
    ModelManager modelManager = new ModelManager();
    private Stage stage;
    private double[][] densePCAData;
    private R[] transformedData;
    private int[] labels;
    private ObservableList<GridSearch.GridSearchResult> gridSearchResults;
    private R[] rawData;
    private ForkJoinPool forkJoinPool = new ForkJoinPool(2);
    private Scaler scaler;
    private Projection<R> projection;
    private String smileModel;
    private double threshold = 0.5;
    private Node activePoint;
    private TrainedBinarySmileModel currentModel;

    public void exportToCSV() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save CSV");
        File file = fileChooser.showSaveDialog(stage);
        if (file != null) {
            final Table vc;
            if (usePCA()) {
                List<String> names = new ArrayList<>();
                List<FeatureExtractor.FeatureType> featureTypes = new ArrayList<>();
                for (int i = 0; i < Integer.parseInt(pcaDimensions.getText()); i++) {
                    names.add("Transformed-Dim" + i);
                    featureTypes.add(FeatureExtractor.FeatureType.Continuous);
                }
                vc = TablesawConverter.toTable("VC", names.toArray(new String[0]), featureTypes.toArray(new FeatureExtractor.FeatureType[0]), transformedData);
            } else {
                vc = TablesawConverter.toTable("VC", getFeatureExtractor().getFeatureNames(), getFeatureExtractor().getFeatureTypes(), transformedData);
            }
            vc.addColumns(BooleanColumn.create("Click", Arrays.stream(labels).mapToObj(i -> i == 1).toArray(Boolean[]::new)));
            CsvWriter writer = null;
            try {
                writer = new CsvWriter(vc, CsvWriteOptions.builder(file).header(true).build());
                writer.write();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    public void fetchData(ActionEvent actionEvent) {
        fetchButton.setDisable(true);
        fetchProgress.setVisible(true);
        fetchProgress.setProgress(0);
        fetchResult.setText("");

        forkJoinPool.execute(() -> {
            TrainingDataSet<R> trainingDataSet = null;
            try {
                trainingDataSet = getTrainingData(Double.parseDouble(samplingFactor.getText()), Long.parseLong(limitData.getText()),
                        progress -> Platform.runLater(() -> fetchProgress.setProgress(progress)));
                rawData = trainingDataSet.getFeatures();
                labels = trainingDataSet.getLabels();
                Platform.runLater(() -> {
                    fetchProgress.setProgress(1);
                    accordion.getPanes().get(1).setDisable(false);
                    accordion.getPanes().get(1).setExpanded(true);
                    fetchResult.setText(String.format("fetched %s datums", rawData.length));
                    fetchButton.setDisable(false);
                });
            } catch (Exception e) {
                e.printStackTrace();
                Platform.runLater(() -> fetchResult.setText("Error"));
                fetchButton.setDisable(false);
            }
        });
    }

    public abstract Pane getDataInput();

    public abstract Class<R> getFeatureDataType();

    public abstract FeatureExtractor<T, R> getFeatureExtractor();

    public void gridSearch(ActionEvent actionEvent) {
        gridSearchButton.setDisable(true);
        trainingProgress.setProgress(0.0);
        trainingProgress.setVisible(true);
        modelColumn.setCellValueFactory(param -> Bindings.createStringBinding(() -> param.getValue().getMLModelType().name()));
        paramsColumn.setCellValueFactory(param -> Bindings
                .createStringBinding(() -> param.getValue().getParams().entrySet().stream().map(e -> e.getKey() + ":" + e.getValue()).collect(joining("|"))));
        accuracyColumn.setCellValueFactory(param -> new SimpleDoubleProperty(param.getValue().getcVresult().getMeasure("Accuracy")).asObject());
        recallColumn.setCellValueFactory(param -> new SimpleDoubleProperty(param.getValue().getcVresult().getMeasure("Sensitivity")).asObject());
        precisionColumn.setCellValueFactory(param -> new SimpleDoubleProperty(param.getValue().getcVresult().getMeasure("Precision")).asObject());
        mccScoreColumn.setCellValueFactory(param -> new SimpleDoubleProperty(param.getValue().getcVresult().getMeasure("MCCMeasure")).asObject());
        f1scoreColumn.setCellValueFactory(param -> new SimpleDoubleProperty(param.getValue().getcVresult().getMeasure("FMeasure")).asObject());

        EnumSet<GridSearch.MLModelType> modelTypes = EnumSet.noneOf(GridSearch.MLModelType.class);
        if (SVM_Gaussian.isSelected()) {
            modelTypes.add(GridSearch.MLModelType.SVM_Gaussian);
        }
        if (SVM_Linear.isSelected()) {
            modelTypes.add(GridSearch.MLModelType.SVM_Linear);
        }
        if (SVM_Laplacian.isSelected()) {
            modelTypes.add(GridSearch.MLModelType.SVM_Laplacian);
        }
        if (AdaBoost.isSelected()) {
            modelTypes.add(GridSearch.MLModelType.AdaBoost);
        }
        if (RandomForest.isSelected()) {
            modelTypes.add(GridSearch.MLModelType.RandomForest);
        }
        if (GradientBoostedTree.isSelected()) {
            modelTypes.add(GridSearch.MLModelType.GradientBoostedTree);
        }

        forkJoinPool.execute(() -> {
            if (usePCA() && transformedData == null) {
                GridSearch<double[]> search = new GridSearch<>(modelTypes, 3, double[].class);
                final List<GridSearch.GridSearchResult> list = search.gridSearch(modelTypes, densePCAData, labels, getFeatureExtractor().getNumberOfFeatures(),
                        new ClassificationMeasure[]{new Accuracy(), new Sensitivity(), new Precision(), new MCCMeasure(), new FMeasure()},
                        Integer.parseInt(numThreads.getText()), d -> Platform.runLater(() -> trainingProgress.setProgress((Double) d)));
                gridSearchResults = FXCollections.observableArrayList(list);
            } else {
                GridSearch<R> search = new GridSearch<>(modelTypes, 3, getFeatureDataType());
                final List<GridSearch.GridSearchResult> list = search.gridSearch(modelTypes, transformedData, labels, getFeatureExtractor().getNumberOfFeatures(),
                        new ClassificationMeasure[]{new Accuracy(), new Sensitivity(), new Precision(), new MCCMeasure(), new FMeasure()},
                        Integer.parseInt(numThreads.getText()), d -> Platform.runLater(() -> trainingProgress.setProgress((Double) d)));
                gridSearchResults = FXCollections.observableArrayList(list);
            }

            Platform.runLater(() -> {
                trainingProgress.setProgress(1.0);
                accordion.getPanes().get(3).setDisable(false);
                accordion.getPanes().get(3).setExpanded(true);
                resultRecordTable.setItems(gridSearchResults);
                resultRecordTable.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, newValue) -> showResult(newValue));
                resultRecordTable.getSelectionModel().select(0);
                gridSearchButton.setDisable(false);
            });
        });

    }

    @FXML
    public void initialize() {
        final Label label = new Label("Datatype: " + getFeatureDataType().getSimpleName() + " Features: " + getFeatureExtractor().getNumberOfFeatures());
        label.setStyle("-fx-font-weight: bold");
        dataInput.getChildren().add(label);
        dataInput.getChildren().addAll(getDataInput().getChildren());
        if (getFeatureDataType().equals(SparseArray.class)) {
            ppcaCheckBox.setDisable(true);
            ghaCheckBox.setDisable(true);
            scaleCheckbox.setSelected(false);
            scaleCheckbox.setDisable(true);

            GradientBoostedTree.setDisable(true);
            AdaBoost.setDisable(true);
            RandomForest.setDisable(true);
        }
    }

    public void openModel(ActionEvent actionEvent) {
        resultROC.setVisible(false);
        confusionMatrixVerify.setVisible(false);
        modelInfoLabel.setText("");
        FileChooser fileChooser = new FileChooser();
        File file = fileChooser.showOpenDialog(stage);
        if (file != null) {
            smileModel = file.getPath();
            currentModelLabel.setText(file.getPath());
            if (transformedData != null && labels != null) {
                verifyButton.setDisable(false);

            }
        }

    }

    public void resampleAndScale(ActionEvent actionEvent) {
        sampleAndScaleButton.setDisable(true);
        exportButton.setDisable(true);
        sampleAndScaleProgress.setVisible(true);
        forkJoinPool.execute(() -> {
            if (rawData instanceof double[][]) {
                if (scaleCheckbox.isSelected()) {
                    scaler = new Scaler();
                    scaler.learn((double[][]) rawData);
                    transformedData = (R[]) scaler.transform((double[][]) rawData);
                } else {
                    transformedData = rawData;
                }
                Platform.runLater(() -> sampleAndScaleProgress.setProgress(0.3));
                if (linearkpca.isSelected()) {
                    PCA pca = new PCA((double[][]) transformedData);
                    pca.setProjection(Integer.parseInt(pcaDimensions.getText()));
                    projection = (Projection<R>) pca;
                } else if (ppcaCheckBox.isSelected()) {
                    projection = (Projection<R>) new PPCA((double[][]) transformedData, Integer.parseInt(pcaDimensions.getText()));
                } else if (ghaCheckBox.isSelected()) {
                    projection = (Projection<R>) new GHA(((double[]) transformedData[0]).length, Integer.parseInt(pcaDimensions.getText()), 0.00001);
                } else if (gausskpca.isSelected()) {
                    projection = (Projection<R>) new KPCA<>((double[][]) transformedData, new GaussianKernel(0.5), Integer.parseInt(pcaDimensions.getText()));
                } else if (htangentkpca.isSelected()) {
                    projection = (Projection<R>) new KPCA<>((double[][]) transformedData, new HyperbolicTangentKernel(), Integer.parseInt(pcaDimensions.getText()));
                }
                if (projection != null) {
                    transformedData = (R[]) projection.project(transformedData);
                }
            } else if (rawData instanceof SparseArray[]) {
                transformedData = rawData;
                if (linearkpca.isSelected()) {
                    projection = (Projection<R>) new KPCA<>((SparseArray[]) transformedData, new SparseLinearKernel(), Integer.parseInt(pcaDimensions.getText()));
                } else if (gausskpca.isSelected()) {
                    projection = (Projection<R>) new KPCA<>((SparseArray[]) transformedData, new SparseGaussianKernel(0.5), Integer.parseInt(pcaDimensions.getText()));
                } else if (htangentkpca.isSelected()) {
                    projection = (Projection<R>) new KPCA<>((SparseArray[]) transformedData, new SparseHyperbolicTangentKernel(),
                            Integer.parseInt(pcaDimensions.getText()));
                }
                if (projection != null) {
                    densePCAData = projection.project(transformedData);
                    transformedData = null;
                }
            }

            Platform.runLater(() -> sampleAndScaleProgress.setProgress(0.6));
            int conversion = (int) ((Arrays.stream(labels).average().getAsDouble() * labels.length));
            int noConversion = labels.length - conversion;
            Platform.runLater(() -> {
                sampleAndScaleProgress.setProgress(1.0);
                accordion.getPanes().get(2).setDisable(false);
                classesBarChart.setTitle("Class Distribution");
                classesBarChart.setVisible(true);
                classesBarChart.getData().clear();

                XYChart.Series<String, Number> series = new XYChart.Series<>();
                XYChart.Data<String, Number> data1 = new XYChart.Data<>("Positive", conversion);
                XYChart.Data<String, Number> data2 = new XYChart.Data<>("Negative", noConversion);
                series.getData().add(data1);
                series.getData().add(data2);
                series.setName("Counts");

                classesBarChart.setLegendVisible(false);
                classesBarChart.setVisible(true);
                classesBarChart.getData().add(series);
                exportButton.setVisible(true);
                exportButton.setDisable(false);

                sampleAndScaleButton.setDisable(false);
            });
        });

    }

    public void saveModel(ActionEvent actionEvent) {
        ((Button) actionEvent.getSource()).setDisable(true);
        forkJoinPool.execute(() -> {
            GridSearch.GridSearchResult selectedItem = resultRecordTable.getSelectionModel().getSelectedItem();
            TrainedBinarySmileModel<R> smileModel;
            if (usePCA() && transformedData == null) {
                SmileModelTrainer<double[]> smileModelTrainer = new SmileModelTrainer<>(selectedItem.getClassifierTrainer());
                smileModel = new TrainedBinarySmileModel<>(smileModelTrainer.trainModel(densePCAData, labels), scaler, projection, threshold);
            } else {
                SmileModelTrainer<R> smileModelTrainer = new SmileModelTrainer<>(selectedItem.getClassifierTrainer());
                smileModel = new TrainedBinarySmileModel<>(smileModelTrainer.trainModel(transformedData, labels), scaler, projection, threshold);
            }
            try {
                this.smileModel = modelManager.saveModel(getModelIdentifier(), smileModel, System.currentTimeMillis());
            } catch (IOException e) {
                e.printStackTrace();
            }
            Platform.runLater(() -> {
                ((Button) actionEvent.getSource()).setDisable(false);
                currentModelLabel.setText(this.smileModel);
                verifyButton.setDisable(false);
            });

        });

    }

    public void setStage(Stage primaryStage) {
        stage = primaryStage;
    }

    public void stopProcessing() {
        forkJoinPool.shutdown();
        forkJoinPool = new ForkJoinPool(2);
        gridSearchButton.setDisable(false);
    }

    public void updateThreshold(ActionEvent actionEvent) {
        if (currentModel != null) {
            currentModel.setThreshold(threshold);
            try {
                smileModel = modelManager.saveModel(getModelIdentifier(), currentModel, System.currentTimeMillis());
                verifyModel(null);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void verifyModel(ActionEvent actionEvent) {
        if (rawData == null || labels == null) {
            noDataLabel.setVisible(true);
        } else {
            verifyButton.setDisable(true);
            noDataLabel.setVisible(false);
            forkJoinPool.execute(() -> {
                TrainedBinarySmileModel<R> model = null;
                try {
                    model = PersistenceUtils.deserialize(Paths.get(smileModel));
                } catch (IOException | ClassNotFoundException e) {
                    e.printStackTrace();
                    return;
                }

                double[] preds = new double[labels.length];
                for (int i = 0; i < rawData.length; i++) {
                    preds[i] = model.predict(rawData[i]).getPosteriori()[1];
                }
                RocCurve rocCurve = new RocCurve(labels, preds);
                TrainedBinarySmileModel finalModel = model;
                final ConfusionMatrix matrix = new ConfusionMatrix(labels, Arrays.stream(preds).mapToInt(d -> d > finalModel.getThreshold() ? 1 : 0).toArray());
                threshold = finalModel.getThreshold();
                currentModel = finalModel;

                Platform.runLater(() -> {
                    resultROC.setVisible(true);
                    resultROC.getData().clear();
                    addROC(resultROC, "ROC", rocCurve, true, finalModel.getThreshold());
                    modelInfoLabel
                            .setText(String.format("Classifier: %s\nProjection: %s\nScaling: %s\nThreshold: %s", finalModel.getClassifier().getClass().getSimpleName(),
                                    finalModel.getProjection() != null ? finalModel.getProjection().getClass().getSimpleName() : "None",
                                    finalModel.getScaler() != null ? finalModel.getScaler().getClass().getSimpleName() : "None", finalModel.getThreshold()));

                    placeholderColumnVerify.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getLeft()));
                    noConversionColumnVerify.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getMiddle()));
                    conversionColumnVerify.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getRight()));

                    Triple<String, String, String> row1 = Triple.of("Negative", String.valueOf(matrix.getMatrix()[0][0]), String.valueOf(matrix.getMatrix()[0][1]));
                    Triple<String, String, String> row2 = Triple.of("Positive", String.valueOf(matrix.getMatrix()[1][0]), String.valueOf(matrix.getMatrix()[1][1]));
                    ObservableList<Triple<String, String, String>> list = FXCollections.observableArrayList(row1, row2);
                    confusionMatrixVerify.setItems(list);
                    confusionMatrixVerify.setVisible(true);
                    verifyButton.setDisable(false);
                });

            });
        }

    }

    protected abstract String getModelIdentifier();

    protected abstract TrainingDataSet<R> getTrainingData(double resamplingRate, long limit, Consumer<Double> callback) throws Exception;

    private void addROC(LineChart<Number, Number> resultROC, String name, RocCurve rocCurve, boolean clickable, double threshold) {
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(name);
        resultROC.getData().add(series);

        final double[] truePositiveRates = rocCurve.getTruePositiveRates();
        final double[] falsePositiveRates = rocCurve.getFalsePositiveRates();
        final double[] thresholds = rocCurve.getThresholds();

        for (int i = 0; i < truePositiveRates.length; i++) {
            XYChart.Data<Number, Number> data = new XYChart.Data<>(falsePositiveRates[i], truePositiveRates[i], thresholds[i]);
            series.getData().add(data);
            if (thresholds[i] == threshold) {
                setActivePoint(data.getNode());
            }

            if (clickable) {
                data.getNode().setOnMouseClicked(event -> {
                    setActivePoint(data.getNode());
                    setThreshold((Double) data.getExtraValue());
                });

            }

        }

    }

    private void setActivePoint(Node node) {
        if (activePoint != null) {
            activePoint.setStyle("");
        }
        activePoint = node;
        activePoint.setStyle("-fx-background-color: blue;");

    }

    private void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    private void showResult(GridSearch.GridSearchResult gridSearchResult) {
        if (gridSearchResult != null) {
            final ConfusionMatrix matrix = gridSearchResult.getcVresult().getConfusionMatrix();
            if (matrix != null) {
                placeholderColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getLeft()));
                negativeColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getMiddle()));
                positiveColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getRight()));

                Triple<String, String, String> row1 = Triple.of("Negative", String.valueOf(matrix.getMatrix()[0][0]), String.valueOf(matrix.getMatrix()[0][1]));
                Triple<String, String, String> row2 = Triple.of("Positive", String.valueOf(matrix.getMatrix()[1][0]), String.valueOf(matrix.getMatrix()[1][1]));
                ObservableList<Triple<String, String, String>> list = FXCollections.observableArrayList(row1, row2);
                confusionMatrix.setItems(list);
                confusionMatrix.setVisible(true);
            } else {
                confusionMatrix.setVisible(false);
            }
            rocCurve.setVisible(true);
            rocCurve.getData().clear();
            addROC(rocCurve, "ROC", gridSearchResult.getcVresult().getRoc(), true, threshold);
        }

    }

    private boolean usePCA() {
        return linearkpca.isSelected() || ppcaCheckBox.isSelected() || ghaCheckBox.isSelected() || gausskpca.isSelected() || htangentkpca.isSelected();
    }
}