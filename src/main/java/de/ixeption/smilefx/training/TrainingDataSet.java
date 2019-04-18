package de.ixeption.smilefx.training;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import smile.feature.FeatureTransform;
import smile.projection.Projection;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class TrainingDataSet<T> {

    private final List<T> _features;
    private final TIntList _labels;
    private double[][] _processedFeatures;
    private Class<?> _clazz;
    private boolean _projected;
    private boolean _scaled;


    public TrainingDataSet(Class<T> clazz) {
        _clazz = clazz;
        _features = new ArrayList<>();
        _labels = new TIntArrayList();
    }

    public TrainingDataSet(T[] x, int[] labels, Class<T> clazz) {
        _features = Arrays.asList(x);
        _labels = new TIntArrayList(labels);
        _clazz = clazz;
    }

    public void addDatapoint(T value, int label) throws IllegalArgumentException {
        if (value instanceof double[]) {
            if (!_features.isEmpty() && ((List<double[]>) _features).get(0).length != ((double[]) value).length) {
                throw new IllegalArgumentException("Inconsistent vector size");
            }
        }

        _features.add(value);
        _labels.add(label);
    }

    public <R> R[] getFeatures() {
        // Java sucks here
        if (_processedFeatures != null) {
            return (R[]) _processedFeatures;
        }
        final T[] instance = (T[]) Array.newInstance(_clazz, _features.size());
        for (int i = 0; i < _features.size(); i++) {
            instance[i] = _features.get(i);
        }
        return (R[]) instance;
    }

    public int[] getLabels() {
        return _labels.toArray();
    }

    public T[] getRawFeatures() {
        // Java sucks here
        final T[] instance = (T[]) Array.newInstance(_clazz, _features.size());
        for (int i = 0; i < _features.size(); i++) {
            instance[i] = _features.get(i);
        }
        return instance;
    }

    public int getSize() {
        return _features.size();
    }

    public Class<?> getType() {
        return _clazz;
    }

    public boolean isProjected() {
        return _projected;
    }

    public boolean isScaled() {
        return _scaled;
    }

    public void project(Projection<T> projection) {
        _processedFeatures = projection.project(getFeatures());
        _projected = true;
    }

    public void resetScalerAndProjection() {
        _scaled = false;
        _projected = false;
        _processedFeatures = null;
    }

    public void scale(FeatureTransform featureTransform) {
        _processedFeatures = featureTransform.transform((double[][]) getRawFeatures());
        _scaled = true;
    }
}
