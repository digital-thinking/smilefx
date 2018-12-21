package de.ixeption.smilefx.util;

import gnu.trove.list.array.TDoubleArrayList;
import smile.validation.Fallout;
import smile.validation.Sensitivity;

import java.util.Arrays;


public class RocCurve {

    private final double[] truePositiveRates;
    private final double[] falsePositiveeRates;
    private final double[] thresholds;
    private Sensitivity sensitivity = new Sensitivity();
    private Fallout fallout = new Fallout();


    public RocCurve(int[] labels, double[] prediction) {
        TDoubleArrayList tprs = new TDoubleArrayList();
        TDoubleArrayList fprs = new TDoubleArrayList();
        TDoubleArrayList ts = new TDoubleArrayList();
        for (double t = 0.01; t < 1.00; t += 0.01) {
            double finalT = t;
            final int[] pred = Arrays.stream(prediction).mapToInt(p -> p > finalT ? 1 : 0).toArray();
            tprs.add(sensitivity.measure(labels, pred));
            fprs.add(fallout.measure(labels, pred));
            ts.add(t);
        }
        thresholds = ts.toArray();
        truePositiveRates = tprs.toArray();
        falsePositiveeRates = fprs.toArray();
    }

    public double[] getFalsePositiveRates() {
        return falsePositiveeRates;
    }

    public double[] getThresholds() {
        return thresholds;
    }

    public double[] getTruePositiveRates() {
        return truePositiveRates;
    }
}
