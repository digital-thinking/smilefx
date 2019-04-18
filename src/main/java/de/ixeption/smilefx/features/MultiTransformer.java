package de.ixeption.smilefx.features;

import smile.data.Attribute;
import smile.feature.FeatureTransform;


public class MultiTransformer extends FeatureTransform {

    private FeatureTransform _featureTransform1;
    private FeatureTransform _featureTransform2;


    public MultiTransformer(FeatureTransform featureTransform1, FeatureTransform featureTransform2) {
        _featureTransform1 = featureTransform1;
        _featureTransform2 = featureTransform2;
    }

    @Override
    public void learn(Attribute[] attributes, double[][] data) {
        _featureTransform1.learn(attributes, data);
        double[][] transform = _featureTransform1.transform(data);
        _featureTransform2.learn(attributes, transform);
    }

    @Override
    public void learn(double[][] data) {
        _featureTransform1.learn(data);
        double[][] transform = _featureTransform1.transform(data);
        _featureTransform2.learn(transform);
    }

    @Override
    public double[] transform(double[] x) {
        return _featureTransform2.transform(_featureTransform1.transform(x));
    }
}
