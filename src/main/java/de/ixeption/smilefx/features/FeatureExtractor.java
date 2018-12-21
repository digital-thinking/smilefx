package de.ixeption.smilefx.features;

public interface FeatureExtractor<T, R> {

    R extract(T value);

    String[] getFeatureNames();

    String getFeatureNameForIndex(int index);

    FeatureType[] getFeatureTypes();

    int getNumberOfFeatures();

    enum FeatureType {
        Binary, Continuous
    }
}
