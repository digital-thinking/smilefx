package de.ixeption.smilefx.example;

import de.ixeption.smilefx.controller.AbstractController;
import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.features.GenericFeatureExtractorBuilder;
import de.ixeption.smilefx.training.TrainingDataSet;
import javafx.scene.layout.Pane;

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
        return builder.build();
    }

    @Override
    protected String getModelIdentifier() {
        return "MyModel";
    }

    @Override
    protected TrainingDataSet<double[]> getTrainingData(double resampleRate, long limit, Consumer<Double> callback) {
        final TrainingDataSet<double[]> trainingDataSet = new TrainingDataSet<>();
        trainingDataSet.addDatapoint(new double[]{0, 1}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1}, 0);
        trainingDataSet.addDatapoint(new double[]{0, 1}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1}, 0);
        trainingDataSet.addDatapoint(new double[]{0, 1}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1}, 0);
        trainingDataSet.addDatapoint(new double[]{0, 1}, 1);
        trainingDataSet.addDatapoint(new double[]{1, 0}, 1);
        trainingDataSet.addDatapoint(new double[]{0, 0}, 0);
        trainingDataSet.addDatapoint(new double[]{1, 1}, 0);
        return trainingDataSet;
    }
}
