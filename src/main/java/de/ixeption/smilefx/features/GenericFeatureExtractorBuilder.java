package de.ixeption.smilefx.features;

import java.util.ArrayList;
import java.util.EnumSet;


public class GenericFeatureExtractorBuilder<T> {

    private final ArrayList<String> _features = new ArrayList<>();
    private final ArrayList<SerializableFunction<T, Double>> _extractors = new ArrayList<>();
    private final ArrayList<FeatureExtractor.FeatureType> _featureTypes = new ArrayList<>();


    public GenericFeatureExtractorBuilder<T> addEnumFeature(String feature, Class<? extends Enum> aClass, SerializableFunction<T, Enum> enumGet) {
        final Enum[] constants = aClass.getEnumConstants();
        for (Enum e : constants) {
            addFeature(feature + "_" + e.name(), t -> (e == enumGet.apply(t) ? 1.0 : 0.0), FeatureExtractor.FeatureType.Binary);
        }
        return this;
    }

    public GenericFeatureExtractorBuilder<T> addEnumSetFeature(String feature, Class<? extends Enum> aClass, SerializableFunction<T, EnumSet> enumSetGet) {
        final Enum[] constants = aClass.getEnumConstants();
        for (Enum e : constants) {
            addFeature(feature + "_" + e.name(), t -> {
                final EnumSet value = enumSetGet.apply(t);
                return (value == null ? 0 : value.contains(e) ? 1.0 : 0.0);
            }, FeatureExtractor.FeatureType.Binary);
        }
        return this;
    }

    public GenericFeatureExtractorBuilder<T> addFeature(String feature, SerializableFunction<T, Double> extract, FeatureExtractor.FeatureType featureType) {
        _features.add(feature);
        _extractors.add(extract);
        _featureTypes.add(featureType);
        return this;
    }

    public GenericFeatureExtractorBuilder<T> addOneHotFeature(String[] features, SerializableFunction<T, String> extract) {
        for (String feature : features) {
            addFeature(feature, t -> extract.apply(t).equals(feature) ? 1.0 : 0.0, FeatureExtractor.FeatureType.Binary);
        }
        return this;

    }

    public GenericFeatureExtractor<T> build() {
        return new GenericFeatureExtractor<>(_features, _extractors, _featureTypes);
    }

}
