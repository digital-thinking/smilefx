package de.ixeption.smilefx.example;

import de.ixeption.smilefx.controller.AbstractController;
import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.features.GenericFeatureExtractorBuilder;
import de.ixeption.smilefx.training.GridSearch;
import de.ixeption.smilefx.training.TrainingDataSet;
import javafx.scene.layout.Pane;
import smile.classification.SVM;
import smile.math.kernel.LinearKernel;
import smile.stat.distribution.GaussianDistribution;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Random;
import java.util.function.Consumer;


public class SmileFXController extends AbstractController<double[], double[]> {

    @Override
    public Pane getDataInput() {
        final Pane pane = new Pane();
        return pane;

    }

    @Override
    public Class<double[]> getFeatureDataType() {
        return double[].class;
    }

    @Override
    public FeatureExtractor<double[], double[]> getFeatureExtractor() {
        final GenericFeatureExtractorBuilder<double[]> builder = new GenericFeatureExtractorBuilder<>();
        builder.addFeature("First", doubles -> doubles[0], FeatureExtractor.FeatureType.Binary);
        builder.addFeature("Second", doubles -> doubles[1], FeatureExtractor.FeatureType.Binary);
        builder.addFeature("Third", doubles -> doubles[2], FeatureExtractor.FeatureType.Continuous);
        return builder.build();
    }

    @Override
    public Pane getPredictInput() {
        return new Pane();
    }

    @Override
    protected String getModelIdentifier() {
        return "MyModel";
    }

    @Override
    protected TrainingDataSet<double[]> getTrainingData(double resampleRate, long limit, Consumer<Double> callback) {
        final TrainingDataSet<double[]> trainingDataSet = new TrainingDataSet<>(double[].class);
        GaussianDistribution normalDisNeg = new GaussianDistribution(1, 0.5);
        GaussianDistribution normalDisPos = new GaussianDistribution(1.5, 0.8);
        Random random = new Random();
        limit = limit < 0 ? 10000 : limit;
        for (int i = 0; i < limit; i += 2) {
            trainingDataSet.addDatapoint(new double[]{random.nextInt(2), random.nextInt(2), normalDisPos.rand()}, 1);
            if (random.nextDouble() < resampleRate) {
                trainingDataSet.addDatapoint(new double[]{random.nextInt(2), random.nextInt(2), normalDisNeg.rand()}, 0);
            }
        }

        return trainingDataSet;
    }

    @Override
    protected TrainingDataSet<double[]> getValidationData() throws Exception {
        final TrainingDataSet<double[]> trainingDataSet = new TrainingDataSet<>(double[].class);
        GaussianDistribution normalDisNeg = new GaussianDistribution(1, 0.5);
        GaussianDistribution normalDisPos = new GaussianDistribution(1.5, 0.8);
        Random random = new Random();
        for (int i = 0; i < 100; i += 2) {
            trainingDataSet.addDatapoint(new double[]{random.nextInt(2), random.nextInt(2), normalDisPos.rand()}, 1);
            trainingDataSet.addDatapoint(new double[]{random.nextInt(2), random.nextInt(2), normalDisNeg.rand()}, 0);
        }

        return trainingDataSet;
    }

    public static class SingleSVMGridSearch extends GridSearch<double[]> {

        public SingleSVMGridSearch(EnumSet<MLModelType> mLModelTypeToSearches, int foldk) {
            super(mLModelTypeToSearches, foldk, double[].class);
        }

        @Override
        protected void gridSearchModel(MLModelType model, double mean, int numberOfFeatures, Class<double[]> type) {
            switch (model) {
                case SVM_Linear:
                    SVM.Trainer<double[]> trainer = new SVM.Trainer<>(new LinearKernel(), 0.3, 0.9);
                    HashMap<String, String> params = new HashMap<>();
                    params.put("CP", String.valueOf(0.3));
                    params.put("CN", String.valueOf(0.9));
                    addToCrossValidation(model, trainer, params);
            }

        }
    }
}
