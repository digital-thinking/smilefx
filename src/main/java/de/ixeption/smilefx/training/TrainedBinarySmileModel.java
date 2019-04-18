package de.ixeption.smilefx.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.classification.*;
import smile.feature.FeatureTransform;
import smile.math.SparseArray;
import smile.projection.Projection;

import javax.annotation.Nonnull;
import java.io.Serializable;
import java.util.Optional;


public class TrainedBinarySmileModel<T> implements Serializable {

    private static final long serialVersionUID = -761791491540444400L;
    private static final Logger _log = LoggerFactory.getLogger(TrainedBinarySmileModel.class);
    private final FeatureTransform _scaler;
    private final SoftClassifier _classifier;
    private final Projection<T> _projection;
    private double _threshold;


    public TrainedBinarySmileModel(@Nonnull SoftClassifier classifier, FeatureTransform scaler, Projection<T> projection, double threshold) {
        _scaler = scaler;
        _classifier = classifier;
        _projection = projection;
        _threshold = threshold;
    }

    public Classifier getClassifier() {
        return _classifier;
    }

    public Optional<double[]> getImportancesIfAvailable() {
        if (_classifier instanceof DecisionTree) {
            return Optional.of(((DecisionTree) _classifier).importance());
        }
        if (_classifier instanceof RandomForest) {
            return Optional.of(((RandomForest) _classifier).importance());
        }
        if (_classifier instanceof AdaBoost) {
            return Optional.of(((AdaBoost) _classifier).importance());
        }
        if (_classifier instanceof GradientTreeBoost) {
            return Optional.of(((GradientTreeBoost) _classifier).importance());
        }
        return Optional.empty();
    }

    public Projection<T> getProjection() {
        return _projection;
    }

    public FeatureTransform getScaler() {
        return _scaler;
    }

    public double getThreshold() {
        return _threshold;
    }

    public void setThreshold(double threshold) {
        _threshold = threshold;
    }

    public PredictionWithThreshold predict(T x) {
        if (x instanceof double[]) {
            if (_scaler != null)
                x = (T) _scaler.transform((double[]) x);
            if (_projection != null)
                x = (T) _projection.project(x);

            final double[] posteriori = new double[2];
            final int predict = _classifier.predict(x, posteriori);
            return new PredictionWithThreshold(posteriori, _threshold);
        } else if (x instanceof SparseArray) {
            if (_projection != null) {
                double[] projected = _projection.project(x);
                final double[] posteriori = new double[2];
                final int predict = _classifier.predict(projected, posteriori);
                return new PredictionWithThreshold(posteriori, _threshold);
            } else {
                final double[] posteriori = new double[2];
                final int predict = _classifier.predict(x, posteriori);
                return new PredictionWithThreshold(posteriori, _threshold);
            }
        } else {
            _log.error("Invalid feature class: " + x.getClass());
            return null;

        }

    }

}
