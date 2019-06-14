package de.ixeption.smilefx.controller;

import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.features.MultiTransformer;
import de.ixeption.smilefx.training.*;
import de.ixeption.smilefx.util.*;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.chart.*;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jetbrains.annotations.NotNull;
import smile.feature.FeatureTransform;
import smile.feature.RobustStandardizer;
import smile.feature.Scaler;
import smile.math.Histogram;
import smile.math.Math;
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
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;


public abstract class AbstractController<T, R> {
    public TextField samplingFactor;
    public Label fetchResult;
    public Accordion accordion;
    public TextField numThreads;
    public Pane featureDistribution;
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
    public BarChart<String, Number> corrBarChart;
    public LineChart<Number, Number> rocCurve;
    public TextField limitData;
    public TableView<Triple<String, String, String>> confusionMatrix;
    public TableColumn<Triple<String, String, String>, String> placeholderColumn;
    public TableColumn<Triple<String, String, String>, String> negativeColumn;
    public TableColumn<Triple<String, String, String>, String> positiveColumn;
    public Button exportButton;
    public CheckBox scaleCheckbox;
    public CheckBox standardizeCheckbox;
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
    public LineChart<Number, Number> finalROC;
    public TableView<Triple<String, String, String>> confusionMatrixVerify;
    public TableColumn<Triple<String, String, String>, String> placeholderColumnVerify;
    public TableColumn<Triple<String, String, String>, String> noConversionColumnVerify;
    public TableColumn<Triple<String, String, String>, String> conversionColumnVerify;
    public Button fetchButton;
    public Button sampleAndScaleButton;
    public Label modelInfoLabel;
    public Button updateThreshold;
    public HBox predictPane;
    public TableView<Pair<String, FeatureExtractor.FeatureType>> featuresTable;
    public TableColumn<Pair<String, FeatureExtractor.FeatureType>, String> featureNameColumn, featureTypeColumn;
    public TableView<Pair<String, Double>> featureImportances;
    public TableColumn<Pair<String, Double>, String> featureImportancesName;
    public TableColumn<Pair<String, Double>, Double> featureImportancesValue;
    public TextField kFolds;
    public Label extractorInfo;
    public Label featureInfo;
    ModelManager modelManager = new ModelManager();
    private Stage stage;
    private ObservableList<GridSearch.GridSearchResult> gridSearchResults;
    private ForkJoinPool forkJoinPool = new ForkJoinPool(1);

    private FeatureTransform featureTransform;
    private Projection<R> projection;
    private String modelPath;
    private double threshold = 0.5;
    private Set<Node> activePoints = new HashSet<>();
    private TrainedBinarySmileModel currentModel;
    private TrainingDataSet<R> validationData;
    private TrainingDataSet<R> trainingDataSet;

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
                    names.add("Projection_" + i);
                    featureTypes.add(FeatureExtractor.FeatureType.Continuous);
                }
                vc = TablesawConverter.toTable("VC", names.toArray(new String[0]), featureTypes.toArray(new FeatureExtractor.FeatureType[0]),
                        trainingDataSet.getFeatures());
            } else {
                vc = TablesawConverter.toTable("VC", getFeatureExtractor().getFeatureNames(), getFeatureExtractor().getFeatureTypes(),
                        trainingDataSet.getFeatures());
            }
            vc.addColumns(BooleanColumn.create("Label", Arrays.stream(trainingDataSet.getLabels()).mapToObj(i -> i == 1).toArray(Boolean[]::new)));

            try {
                new CsvWriter().write(vc, CsvWriteOptions.builder(file).header(true).build());
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
            try {
                trainingDataSet = getTrainingData(Double.parseDouble(samplingFactor.getText()), Long.parseLong(limitData.getText()),
                        progress -> Platform.runLater(() -> fetchProgress.setProgress(progress)));
                Platform.runLater(() -> {
                    fetchProgress.setProgress(1);
                    accordion.getPanes().get(1).setDisable(false);
                    fetchResult.setText(String.format("fetched %s datums", trainingDataSet.getSize()));
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

    public abstract Pane getPredictInput();

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
            GridSearch<R> search = getGridSearch(modelTypes);
            search.printFeatureImportance(trainingDataSet.getLabels(), trainingDataSet.getFeatures(), 20, getFeatureExtractor());
            final List<GridSearch.GridSearchResult> list = search.gridSearch(modelTypes, trainingDataSet, getFeatureExtractor().getNumberOfFeatures(),
                    new ClassificationMeasure[]{new Accuracy(), new Sensitivity(), new Precision(), new MCCMeasure(), new FMeasure()},
                    Integer.parseInt(numThreads.getText()), d -> Platform.runLater(() -> trainingProgress.setProgress((Double) d)));
            gridSearchResults = FXCollections.observableArrayList(list);

            Platform.runLater(() -> {
                predictPane.setVisible(true);
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

    @NotNull
    protected GridSearch<R> getGridSearch(EnumSet<GridSearch.MLModelType> modelTypes) {
        return new GridSearch<>(modelTypes, Integer.parseInt(kFolds.getText()), getFeatureDataType());
    }

    @FXML
    public void initialize() {
        initFeatureTable();
        extractorInfo.setStyle("-fx-font-weight: bold");
        extractorInfo.setText("Datatype: " + getFeatureDataType().getSimpleName() + " Features: " + getFeatureExtractor().getNumberOfFeatures());
        predictPane.getChildren().add(getPredictInput());
        if (getFeatureDataType().equals(SparseArray.class)) {
            ppcaCheckBox.setDisable(true);
            ghaCheckBox.setDisable(true);
            scaleCheckbox.setSelected(false);
            scaleCheckbox.setDisable(true);
            standardizeCheckbox.setDisable(true);
            standardizeCheckbox.setSelected(false);

            GradientBoostedTree.setDisable(true);
            AdaBoost.setDisable(true);
            RandomForest.setDisable(true);
        }
    }

    public void openModel(ActionEvent actionEvent) {
        predictPane.setVisible(true);
        finalROC.setVisible(false);
        confusionMatrixVerify.setVisible(false);
        modelInfoLabel.setText("");
        FileChooser fileChooser = new FileChooser();
        File file = fileChooser.showOpenDialog(stage);
        if (file != null) {
            modelPath = file.getPath();
            currentModelLabel.setText(file.getName());
            try {
                currentModel = PersistenceUtils.deserialize(Paths.get(modelPath));
                threshold = currentModel.getThreshold();
            } catch (IOException | ClassNotFoundException e) {
                e.printStackTrace();
                return;
            }
            if (trainingDataSet.getFeatures() != null && trainingDataSet.getLabels() != null) {
                verifyButton.setDisable(false);
            }
            showImportances((double[]) currentModel.getImportancesIfAvailable().orElseGet(null));
        }

    }

    public void resampleAndScale(ActionEvent actionEvent) {
        sampleAndScaleButton.setDisable(true);
        exportButton.setDisable(true);
        sampleAndScaleProgress.setVisible(true);
        forkJoinPool.execute(() -> {

            trainingDataSet.resetScalerAndProjection();
            if (scaleCheckbox.isSelected() && !standardizeCheckbox.isSelected()) {
                featureTransform = new Scaler(true);
            } else if (!scaleCheckbox.isSelected() && standardizeCheckbox.isSelected()) {
                featureTransform = new RobustStandardizer(true);
            } else if (scaleCheckbox.isSelected() && standardizeCheckbox.isSelected()) {
                featureTransform = new MultiTransformer(new Scaler(true), new RobustStandardizer(true));
            } else {
                featureTransform = null;
            }
            if (featureTransform != null) {
                featureTransform.learn((double[][]) trainingDataSet.getRawFeatures());
                trainingDataSet.scale(featureTransform);
            }
            if (usePCA()) {
                if (trainingDataSet.getType().equals(double[].class)) {
                    setDenseProjection();
                } else {
                    setSparseProjection();
                }
                trainingDataSet.project(projection);
            } else {
                projection = null;
            }

            if (trainingDataSet.getType().equals(double[].class)) {
                double[][] transposed = Math.transpose(trainingDataSet.getFeatures());
                PearsonsCorrelation correlation = new PearsonsCorrelation();
                double[] labelsDouble = Arrays.stream(trainingDataSet.getLabels()).asDoubleStream().toArray();
                List<XYChart.Data<String, Number>> dataList = IntStream.range(0, transposed.length).mapToObj(i -> {
                    if (trainingDataSet.isProjected()) {
                        return Pair.of("Projection_" + i, correlation.correlation(labelsDouble, transposed[i]));
                    }
                    return Pair.of(getFeatureExtractor().getFeatureNames()[i], correlation.correlation(labelsDouble, transposed[i]));
                }).filter(p -> !p.getValue().isNaN()).sorted(Comparator.comparing((Function<Pair<String, Double>, Double>) Pair::getRight).reversed()).limit(10)
                        .map(pair -> new XYChart.Data<>(pair.getKey(), (Number) pair.getValue())).collect(Collectors.toList());

                Platform.runLater(() -> {
                    corrBarChart.setTitle("Correlations");
                    corrBarChart.getData().clear();
                    XYChart.Series<String, Number> series = new XYChart.Series<>();
                    series.getData().addAll(dataList);
                    series.setName("Pearson Correlation");
                    corrBarChart.getData().add(series);
                    corrBarChart.setVisible(true);
                });
            }

            Platform.runLater(() -> sampleAndScaleProgress.setProgress(0.6));
            int conversion = (int) ((Arrays.stream(trainingDataSet.getLabels()).average().getAsDouble() * trainingDataSet.getLabels().length));
            int noConversion = trainingDataSet.getLabels().length - conversion;
            Platform.runLater(() -> {
                sampleAndScaleProgress.setProgress(1.0);
                accordion.getPanes().get(2).setDisable(false);
                classesBarChart.setTitle("Class Distribution");
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
            TrainedBinarySmileModel smileModel;
            if (trainingDataSet.isProjected() || trainingDataSet.isScaled()) {
                SmileModelTrainer<double[]> smileModelTrainer = new SmileModelTrainer<>(selectedItem.getClassifierTrainer());
                smileModel = new TrainedBinarySmileModel<>(smileModelTrainer.trainModel(trainingDataSet.getFeatures(), trainingDataSet.getLabels()),
                        featureTransform, projection, threshold);
            } else {
                SmileModelTrainer<R> smileModelTrainer = new SmileModelTrainer<>(selectedItem.getClassifierTrainer());
                smileModel = new TrainedBinarySmileModel<>(smileModelTrainer.trainModel(trainingDataSet.getFeatures(), trainingDataSet.getLabels()),
                        featureTransform, projection, threshold);
            }
            try {
                modelPath = modelManager.saveModel(getModelIdentifier(), smileModel, System.currentTimeMillis());
                currentModel = smileModel;
            } catch (IOException e) {
                e.printStackTrace();
            }
            Platform.runLater(() -> {
                ((Button) actionEvent.getSource()).setDisable(false);
                currentModelLabel.setText(String.valueOf(Paths.get(this.modelPath).getFileName()));
                verifyButton.setDisable(false);
            });

        });

    }

    public void setDenseProjection() {
        if (linearkpca.isSelected()) {
            PCA pca = new PCA(trainingDataSet.getFeatures());
            pca.setProjection(Integer.parseInt(pcaDimensions.getText()));
            projection = (Projection<R>) pca;
        } else if (ppcaCheckBox.isSelected()) {
            projection = (Projection<R>) new PPCA(trainingDataSet.getFeatures(), Integer.parseInt(pcaDimensions.getText()));
        } else if (ghaCheckBox.isSelected()) {
            projection = (Projection<R>) new GHA(getFeatureExtractor().getNumberOfFeatures(), Integer.parseInt(pcaDimensions.getText()), 0.00001);
        } else if (gausskpca.isSelected()) {
            projection = (Projection<R>) new KPCA<>(trainingDataSet.getFeatures(), new GaussianKernel(0.5), Integer.parseInt(pcaDimensions.getText()));
        } else if (htangentkpca.isSelected()) {
            projection = (Projection<R>) new KPCA<>(trainingDataSet.getFeatures(), new HyperbolicTangentKernel(), Integer.parseInt(pcaDimensions.getText()));
        }
    }

    public void setSparseProjection() {
        if (linearkpca.isSelected()) {
            projection = (Projection<R>) new KPCA<>(trainingDataSet.getFeatures(), new SparseLinearKernel(), Integer.parseInt(pcaDimensions.getText()));
        } else if (gausskpca.isSelected()) {
            projection = (Projection<R>) new KPCA<>(trainingDataSet.getFeatures(), new SparseGaussianKernel(0.5), Integer.parseInt(pcaDimensions.getText()));
        } else if (htangentkpca.isSelected()) {
            projection = (Projection<R>) new KPCA<>(trainingDataSet.getFeatures(), new SparseHyperbolicTangentKernel(), Integer.parseInt(pcaDimensions.getText()));
        }
    }

    public void setStage(Stage primaryStage) {
        stage = primaryStage;
    }

    public void stopProcessing() {
        BalancedCrossValidation.stopAll();
        gridSearchButton.setDisable(false);
    }

    public void updateThreshold(ActionEvent actionEvent) {
        if (currentModel != null) {
            currentModel.setThreshold(threshold);
            try {
                modelPath = modelManager.saveModel(getModelIdentifier(), currentModel, System.currentTimeMillis());
                verifyModel(null);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void verifyModel(ActionEvent actionEvent) {
        if (validationData == null) {
            try {
                validationData = getValidationData();
            } catch (Exception e) {
                System.err.println(e);
                noDataLabel.setText(e.getMessage());
            }
        }

        if (validationData == null || validationData.getFeatures() == null || validationData.getLabels() == null) {
            noDataLabel.setVisible(true);
        } else {
            verifyButton.setDisable(true);
            noDataLabel.setVisible(false);
            forkJoinPool.execute(() -> {
                double[] preds = new double[validationData.getLabels().length];
                for (int i = 0; i < validationData.getFeatures().length; i++) {
                    preds[i] = currentModel.predict(validationData.getRawFeatures()[i]).getPosteriori()[1];
                }
                RocCurve rocCurve = new RocCurve(validationData.getLabels(), preds);
                PrecisionRecallCurve prcCurve = new PrecisionRecallCurve(validationData.getLabels(), preds);
                final ConfusionMatrix matrix = new ConfusionMatrix(validationData.getLabels(),
                        Arrays.stream(preds).mapToInt(d -> d > currentModel.getThreshold() ? 1 : 0).toArray());

                Platform.runLater(() -> {
                    finalROC.setVisible(true);
                    finalROC.getData().clear();
                    addROC(finalROC, "ROC", rocCurve, true, threshold);
                    addPRC(finalROC, "PRC", prcCurve, true, threshold);
                    modelInfoLabel.setText(
                            String.format("Classifier: %s\nProjection: %s\nScaling: %s\nThreshold: %s", currentModel.getClassifier().getClass().getSimpleName(),
                                    currentModel.getProjection() != null ? currentModel.getProjection().getClass().getSimpleName() : "None",
                                    currentModel.getScaler() != null ? currentModel.getScaler().getClass().getSimpleName() : "None", currentModel.getThreshold()));

                    updateConfusionMartix(matrix, placeholderColumnVerify, noConversionColumnVerify, conversionColumnVerify, confusionMatrixVerify);
                    verifyButton.setDisable(false);
                });

            });
        }

    }

    protected TrainedBinarySmileModel getCurrentModel() {
        return currentModel;
    }

    protected abstract String getModelIdentifier();

    protected abstract TrainingDataSet<R> getTrainingData(double resamplingRate, long limit, Consumer<Double> callback) throws Exception;

    protected abstract TrainingDataSet<R> getValidationData() throws Exception;

    private void addBar(double[] values, XYChart.Series<String, Number> seriesPos, int bin) {
        XYChart.Data<String, Number> dataPos = new XYChart.Data<>(String.valueOf(bin),
                Arrays.stream(values).filter(value -> value == bin).count() / (double) values.length);
        seriesPos.getData().add(dataPos);
    }

    private void addPRC(LineChart<Number, Number> resultROC, String name, PrecisionRecallCurve prc, boolean clickable, double threshold) {
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(name);
        resultROC.getData().add(series);

        final double[] x = prc.getRecalls();
        final double[] y = prc.getPrecisions();
        final double[] thresholds = prc.getThresholds();

        drawXYChart(clickable, threshold, series, x, y, thresholds);

    }

    private void addROC(LineChart<Number, Number> resultROC, String name, RocCurve rocCurve, boolean clickable, double threshold) {
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(name);
        resultROC.getData().add(series);

        final double[] x = rocCurve.getFalsePositiveRates();
        final double[] y = rocCurve.getTruePositiveRates();
        final double[] thresholds = rocCurve.getThresholds();

        drawXYChart(clickable, threshold, series, x, y, thresholds);

    }

    private void drawXYChart(boolean clickable, double threshold, XYChart.Series<Number, Number> series, double[] xValues, double[] yValues,
                             double[] thresholds) {
        for (int i = 0; i < yValues.length; i++) {
            XYChart.Data<Number, Number> data = new XYChart.Data<>(xValues[i], yValues[i], thresholds[i]);
            series.getData().add(data);
            if (thresholds[i] == threshold) {
                setActivePoint(data.getNode(), (Double) data.getExtraValue());
            }

            if (clickable) {
                data.getNode().setOnMouseClicked(event -> {
                    setActivePoint(data.getNode(), (Double) data.getExtraValue());
                });

            }

        }
    }

    private void initFeatureTable() {
        featureNameColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getLeft()));
        featureTypeColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getRight().name()));

        String[] names = getFeatureExtractor().getFeatureNames();
        FeatureExtractor.FeatureType[] types = getFeatureExtractor().getFeatureTypes();
        List<Pair<String, FeatureExtractor.FeatureType>> featureData = IntStream.range(0, names.length)//
                .mapToObj(i -> Pair.of(names[i], types[i]))//
                .collect(Collectors.toList());
        featuresTable.setItems(FXCollections.observableArrayList(featureData));
        featuresTable.getSelectionModel().selectedItemProperty().addListener((obs, oldSelection, newSelection) -> {
            if (newSelection != null) {
                showFeatureHist(newSelection.getKey());
            }
        });

    }

    private void setActivePoint(Node node, Double newThreshold) {
        if (!activePoints.isEmpty() && threshold != newThreshold) {
            activePoints.forEach(n -> n.setStyle(""));
        }
        node.setStyle("-fx-background-color: blue;");
        activePoints.add(node);
        threshold = newThreshold;

    }


    private void showBinary(double[] xPos, double[] xNeg) {
        Axis<String> xAxis = new CategoryAxis();
        Axis<Number> yAxis = new NumberAxis();
        BarChart<String, Number> areaChart = new BarChart<>(xAxis, yAxis);
        featureDistribution.getChildren().clear();
        featureDistribution.getChildren().add(areaChart);

        XYChart.Series<String, Number> seriesPos = new XYChart.Series<>();
        XYChart.Series<String, Number> seriesNeg = new XYChart.Series<>();

        for (int i = 0; i < 2; i++) {
            addBar(xPos, seriesPos, i);
            seriesPos.setName("Positive");

            addBar(xNeg, seriesNeg, i);
            seriesNeg.setName("Negative");

        }

        areaChart.setData(FXCollections.observableArrayList(seriesPos, seriesNeg));
        featureDistribution.setVisible(true);
    }

    private void showContinuous(double[] xPos, double[] xNeg, DescriptiveStatistics statisticsPos, DescriptiveStatistics statisticsNeg) {
        Axis<String> xAxis = new CategoryAxis();
        Axis<Number> yAxis = new NumberAxis();
        AreaChart<String, Number> areaChart = new AreaChart<>(xAxis, yAxis);
        areaChart.setCreateSymbols(false);
        featureDistribution.getChildren().clear();
        featureDistribution.getChildren().add(areaChart);

        double percentile05 = Math.min(statisticsPos.getPercentile(5), statisticsNeg.getPercentile(5));
        double percentile95 = Math.max(statisticsPos.getPercentile(95), statisticsNeg.getPercentile(95));
        if (percentile05 == percentile95) {
            featureDistribution.getChildren().clear();
            featureDistribution.getChildren().add(new Label("All values equal " + percentile05));
            return;
        }
        double[] breaks = Histogram.breaks(percentile05, percentile95, 100);

        double[][] histogramPos = Histogram.histogram(xPos, breaks);
        double[][] histogramNeg = Histogram.histogram(xNeg, breaks);

        XYChart.Series<String, Number> seriesPos = new XYChart.Series<>();
        XYChart.Series<String, Number> seriesNeg = new XYChart.Series<>();

        for (int i = 0; i < 100; i++) {
            XYChart.Data<String, Number> dataPos = new XYChart.Data<>(String.format("%.2f", (histogramPos[0][i] + histogramPos[1][i] / 2.0)),
                    histogramPos[2][i] / (double) xPos.length);
            seriesPos.getData().add(dataPos);
            seriesPos.setName("Positive");

            XYChart.Data<String, Number> dataNeg = new XYChart.Data<>(String.format("%.2f", (histogramNeg[0][i] + histogramNeg[1][i] / 2.0)),
                    histogramNeg[2][i] / (double) xNeg.length);
            seriesNeg.getData().add(dataNeg);
            seriesNeg.setName("Negative");

        }

        areaChart.setData(FXCollections.observableArrayList(seriesPos, seriesNeg));
        featureDistribution.setVisible(true);

    }

    private void showFeatureHist(String key) {
        int featureIndex = Arrays.asList(getFeatureExtractor().getFeatureNames()).indexOf(key);
        if (trainingDataSet == null || featureIndex == -1)
            return;
        FeatureExtractor.FeatureType featureType = getFeatureExtractor().getFeatureTypes()[featureIndex];

        int[] labels = trainingDataSet.getLabels();
        R[] rawFeatures = trainingDataSet.getRawFeatures();

        DescriptiveStatistics statisticsPos = new DescriptiveStatistics();
        DescriptiveStatistics statisticsNeg = new DescriptiveStatistics();

        double[] xPos = new double[rawFeatures.length];
        double[] xNeg = new double[rawFeatures.length];

        for (int i = 0; i < rawFeatures.length; i++) {
            R row = rawFeatures[i];
            if (row instanceof double[]) {
                double[] arr = (double[]) row;
                if (labels[i] == 1) {
                    statisticsPos.addValue(arr[featureIndex]);
                    xPos[i] = arr[featureIndex];
                } else {
                    statisticsNeg.addValue(arr[featureIndex]);
                    xNeg[i] = arr[featureIndex];
                }
            }
        }

        StringBuilder builder = new StringBuilder();
        builder.append(String.format("Mean (Pos): %.2f \n", statisticsPos.getMean()));
        builder.append(String.format("Mean (Neg): %.2f \n", statisticsNeg.getMean()));
        builder.append(String.format("Std (Pos): %.2f \n", statisticsPos.getStandardDeviation()));
        builder.append(String.format("Std (Neg): %.2f \n", statisticsNeg.getStandardDeviation()));
        builder.append(String.format("Min (Pos): %.2f \n", statisticsPos.getMin()));
        builder.append(String.format("Min (Neg): %.2f \n", statisticsNeg.getMin()));
        builder.append(String.format("Max (Pos): %.2f \n", statisticsPos.getMax()));
        builder.append(String.format("Max (Neg): %.2f \n", statisticsNeg.getMax()));
        featureInfo.setText(builder.toString());

        if (featureType.equals(FeatureExtractor.FeatureType.Continuous)) {
            showContinuous(xPos, xNeg, statisticsPos, statisticsNeg);
        } else {
            showBinary(xPos, xNeg);
        }

    }

    private void showImportances(double[] importances) {
        if (importances != null) {
            featureImportancesName.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getLeft()));
            featureImportancesValue.setCellValueFactory(param -> new SimpleDoubleProperty(param.getValue().getRight()).asObject());
            List<Pair<String, Double>> pairs;
            if (currentModel.getProjection() == null) {
                pairs = IntStream.range(0, getFeatureExtractor().getNumberOfFeatures())
                        .mapToObj(i -> Pair.of(getFeatureExtractor().getFeatureNames()[i], importances[i])).collect(Collectors.toList());
            } else {
                pairs = IntStream.range(0, importances.length).mapToObj(i -> Pair.of("Projection_" + i, importances[i])).collect(Collectors.toList());
            }
            featureImportances.setItems(FXCollections.observableArrayList(pairs));
            featureImportances.setVisible(true);
        } else {
            featureImportances.setVisible(false);
        }
    }

    private void showResult(GridSearch.GridSearchResult gridSearchResult) {
        if (gridSearchResult != null) {
            final ConfusionMatrix matrix = gridSearchResult.getcVresult().getConfusionMatrix();
            if (matrix != null) {
                updateConfusionMartix(matrix, placeholderColumn, negativeColumn, positiveColumn, confusionMatrix);
            } else {
                confusionMatrix.setVisible(false);
            }
            rocCurve.setVisible(true);
            rocCurve.getData().clear();
            addROC(rocCurve, "ROC", gridSearchResult.getcVresult().getRoc(), true, threshold);
            addPRC(rocCurve, "PRC", gridSearchResult.getcVresult().getPrc(), true, threshold);
        }

    }

    private void updateConfusionMartix(ConfusionMatrix matrix, TableColumn<Triple<String, String, String>, String> placeholderColumn,
                                       TableColumn<Triple<String, String, String>, String> negativeColumn, TableColumn<Triple<String, String, String>, String> positiveColumn,
                                       TableView<Triple<String, String, String>> confusionMatrix) {
        placeholderColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getLeft()));
        negativeColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getMiddle()));
        positiveColumn.setCellValueFactory(param -> new SimpleStringProperty(param.getValue().getRight()));

        Triple<String, String, String> row1 = Triple.of("Negative", String.valueOf(matrix.getMatrix()[0][0]), String.valueOf(matrix.getMatrix()[0][1]));
        Triple<String, String, String> row2 = Triple.of("Positive", String.valueOf(matrix.getMatrix()[1][0]), String.valueOf(matrix.getMatrix()[1][1]));
        ObservableList<Triple<String, String, String>> list = FXCollections.observableArrayList(row1, row2);
        confusionMatrix.setItems(list);
        confusionMatrix.setVisible(true);
    }

    private boolean usePCA() {
        return linearkpca.isSelected() || ppcaCheckBox.isSelected() || ghaCheckBox.isSelected() || gausskpca.isSelected() || htangentkpca.isSelected();
    }
}