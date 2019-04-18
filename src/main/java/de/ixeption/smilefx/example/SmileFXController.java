package de.ixeption.smilefx.example;

import de.ixeption.smilefx.controller.AbstractController;
import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.features.GenericFeatureExtractorBuilder;
import de.ixeption.smilefx.training.TrainingDataSet;
import javafx.scene.layout.Pane;
import smile.stat.distribution.GaussianDistribution;

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
        GaussianDistribution normalDis = new GaussianDistribution(0.5, 1);
        trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
        return trainingDataSet;
    }

    @Override
    protected TrainingDataSet<double[]> getValidationData() throws Exception {
        final TrainingDataSet<double[]> trainingDataSet = new TrainingDataSet<>(double[].class);
        trainingDataSet.addDatapoint(new double[]{0, 1, 0.1}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 1, 0.8}, 0);
        return trainingDataSet;
    }
}
