package de.ixeption.smilefx.training;

import org.apache.commons.lang3.ArrayUtils;
import smile.feature.FeatureTransform;
import smile.projection.Projection;


public class TrainTestSplitDataSet<T> extends TrainingDataSet<T> {

    private final TrainingDataSet<T> _trainingDataSet;
    private final TrainingDataSet<T> _testDataSet;


    public TrainTestSplitDataSet(TrainingDataSet<T> trainingDataSet, TrainingDataSet<T> testDataSet, Class<T> clazz) {
        super(clazz);
        _trainingDataSet = trainingDataSet;
        _testDataSet = testDataSet;
    }

    @Override
    public void addDatapoint(T value, int label) throws IllegalArgumentException {
        throw new RuntimeException("Operation not permitted");
    }

    @Override
    public <R> R[] getFeatures() {
        return ArrayUtils.addAll(_trainingDataSet.getFeatures(), _testDataSet.getFeatures());
    }

    public T[] getFeaturesTest() {
        return _testDataSet.getFeatures();
    }

    public T[] getFeaturesTrain() {
        return _trainingDataSet.getFeatures();
    }

    @Override
    public int[] getLabels() {
        return ArrayUtils.addAll(_trainingDataSet.getLabels(), _testDataSet.getLabels());
    }

    public int[] getLabelsTest() {
        return _testDataSet.getLabels();
    }

    public int[] getLabelsTrain() {
        return _trainingDataSet.getLabels();
    }

    @Override
    public T[] getRawFeatures() {
        return ArrayUtils.addAll(_trainingDataSet.getRawFeatures(), _testDataSet.getRawFeatures());
    }

    @Override
    public int getSize() {
        return _trainingDataSet.getSize() + _testDataSet.getSize();
    }

    @Override
    public boolean isProjected() {
        return _trainingDataSet.isProjected();
    }

    @Override
    public boolean isScaled() {
        return _trainingDataSet.isScaled();
    }

    @Override
    public void project(Projection<T> projection) {
        _trainingDataSet.project(projection);
        _testDataSet.project(projection);
    }

    @Override
    public void resetScalerAndProjection() {
        _trainingDataSet.resetScalerAndProjection();
        _testDataSet.resetScalerAndProjection();
    }

    @Override
    public void scale(FeatureTransform scaler) {
        _trainingDataSet.scale(scaler);
        _testDataSet.scale(scaler);
    }
}
