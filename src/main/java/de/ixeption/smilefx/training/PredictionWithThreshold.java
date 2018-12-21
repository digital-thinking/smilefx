package de.ixeption.smilefx.training;

public class PredictionWithThreshold {

    private final double[] _posteriori;
    private final double _threshold;


    public PredictionWithThreshold(double[] posteriori, double threshold) {
        _posteriori = posteriori;
        _threshold = threshold;
    }

    public int getLabel() {
        return _posteriori[1] > _threshold ? 1 : 0;
    }

    public double[] getPosteriori() {
        return _posteriori;
    }

    public Double getThreshold() {
        return _threshold;
    }
}
