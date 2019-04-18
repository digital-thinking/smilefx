package de.ixeption.smilefx.training;

import de.ixeption.smilefx.util.PrecisionRecallCurve;
import de.ixeption.smilefx.util.RocCurve;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import smile.validation.ConfusionMatrix;


public class CVResult {

    final TObjectDoubleMap<String> _map = new TObjectDoubleHashMap<>();
    final ConfusionMatrix _confusionMatrix;
    final private RocCurve _roc;
    final private PrecisionRecallCurve _prc;


    public CVResult(ConfusionMatrix confusionMatrix, RocCurve roc, PrecisionRecallCurve prc) {
        _confusionMatrix = confusionMatrix;
        _roc = roc;
        _prc = prc;
    }

    public void addMeasure(String classificationMeasure, double value) {
        _map.put(classificationMeasure, value);
    }

    public ConfusionMatrix getConfusionMatrix() {
        return _confusionMatrix;
    }

    public double getMeasure(String measure) {
        return _map.get(measure);
    }

    public PrecisionRecallCurve getPrc() {
        return _prc;
    }

    public double[] getPredictions() {
        return new double[0];
    }

    public RocCurve getRoc() {
        return _roc;
    }

}
