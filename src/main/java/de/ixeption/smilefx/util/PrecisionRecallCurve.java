package de.ixeption.smilefx.util;

import gnu.trove.list.array.TDoubleArrayList;
import smile.validation.Precision;
import smile.validation.Recall;

import java.util.Arrays;


public class PrecisionRecallCurve {

    private final double[] precisions;
    private final double[] recalls;
    private final double[] thresholds;
    private Recall recall = new Recall();
    private Precision precision = new Precision();


    public PrecisionRecallCurve(int[] labels, double[] prediction) {
        TDoubleArrayList prec = new TDoubleArrayList();
        TDoubleArrayList rec = new TDoubleArrayList();
        TDoubleArrayList ts = new TDoubleArrayList();
        for (double t = 0.01; t < 1.00; t += 0.01) {
            double finalT = t;
            final int[] pred = Arrays.stream(prediction).mapToInt(p -> p > finalT ? 1 : 0).toArray();
            prec.add(precision.measure(labels, pred));
            rec.add(recall.measure(labels, pred));
            ts.add(t);
        }
        thresholds = ts.toArray();
        precisions = prec.toArray();
        recalls = rec.toArray();
    }

    public double[] getPrecisions() {
        return precisions;
    }

    public double[] getRecalls() {
        return recalls;
    }

    public double[] getThresholds() {
        return thresholds;
    }

}
