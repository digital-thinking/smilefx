package de.ixeption.smilefx.training;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class TrainingDataSet<T> {

    private final List<T> _features;
    private final TIntList _labels;


    public TrainingDataSet() {
        _features = new ArrayList<>();
        _labels = new TIntArrayList();
    }

    public TrainingDataSet(T[] x, int[] labels) {
        _features = Arrays.asList(x);
        _labels = new TIntArrayList(labels);
    }

    public void addDatapoint(T value, int label) throws IllegalArgumentException {
        // TODO
        //      if ( !_features.isEmpty() && _features.get(0).length != value.length ) {
        //         throw new IllegalArgumentException("Inconsistent vector size");
        //      }
        _features.add(value);
        _labels.add(label);
    }

    public T[] getFeatures() {
        // Java sucks here
        final Class<?> aClass = _features.get(0).getClass();
        final T[] instance = (T[]) Array.newInstance(aClass, _features.size());
        for (int i = 0; i < _features.size(); i++) {
            instance[i] = _features.get(i);
        }
        return instance;
    }

    public int[] getLabels() {
        return _labels.toArray();
    }

    public int getSize() {
        return _features.size();
    }
}
