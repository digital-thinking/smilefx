package de.ixeption.smilefx.features;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.IntStream;


public class GenericFeatureExtractor<T> implements FeatureExtractor<T, double[]>, Serializable {

    private static final long serialVersionUID = -8348881352052583396L;
    private final ArrayList<String> _features;
    private final ArrayList<SerializableFunction<T, Double>> _extractors;
    private final ArrayList<FeatureType> _featureTypes;


    public GenericFeatureExtractor(ArrayList<String> features, ArrayList<SerializableFunction<T, Double>> extractors, ArrayList<FeatureType> featureTypes) {
        _features = features;
        _extractors = extractors;
        _featureTypes = featureTypes;
    }

    @Override
    public double[] extract(T value) {
        final double[] array = IntStream.range(0, _features.size()).mapToDouble(i -> _extractors.get(i).apply(value)).toArray();
        return array;
    }

    @Override
    public String getFeatureNameForIndex(int index) {
        return _features.get(index);
    }

    @Override
    public String[] getFeatureNames() {
        return _features.toArray(new String[0]);
    }

    @Override
    public FeatureType[] getFeatureTypes() {
        return _featureTypes.toArray(new FeatureType[0]);
    }

    @Override
    public int getNumberOfFeatures() {
        return _features.size();
    }
}
