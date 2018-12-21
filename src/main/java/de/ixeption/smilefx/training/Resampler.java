package de.ixeption.smilefx.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class Resampler<T> {

    private static final Logger _log = LoggerFactory.getLogger(Resampler.class);
    private final List<T> _features = new ArrayList<>();
    private final List<Integer> _labels;
    private final Random random = new Random();


    public Resampler(T[] features, int[] labels, double... sampling) {
        final int classes = Arrays.stream(labels).max().orElseThrow(IllegalArgumentException::new) + 1;
        int[] classCounts = new int[classes];
        for (int i : labels) {
            classCounts[i]++;
        }

        if (sampling == null || sampling.length == 0) {
            sampling = new double[classCounts.length];
            int maxcount = Arrays.stream(classCounts).filter(i -> i >= 0).max().orElse(0);
            for (int i = 0; i < classCounts.length; i++) {
                _log.debug("Class {} : {}", i, classCounts[i]);
                sampling[i] = (double) maxcount / classCounts[i];
            }
            _log.info("No sampling rate supplied, using balanced upsampling mode: " + Arrays.toString(sampling));

        }

        if (sampling.length != classCounts.length) {
            throw new IllegalArgumentException(String.format("Classes and sampling rates do not match %s:%s", classCounts.length, sampling.length));
        }

        _labels = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            int times = (int) sampling[labels[i]];
            double frac = sampling[labels[i]] - times;

            for (int j = 0; j < times; j++) {
                _features.add(features[i]);
                _labels.add(labels[i]);
            }

            if (random.nextDouble() <= frac) {
                _features.add(features[i]);
                _labels.add(labels[i]);
            }
        }

    }

    public T[] getFeatures() {
        final Class<?> aClass = _features.get(0).getClass();
        final T[] instance = (T[]) Array.newInstance(aClass, _features.size());
        for (int i = 0; i < _features.size(); i++) {
            instance[i] = _features.get(i);
        }
        return instance;

    }

    public List<Integer> getLabels() {
        return _labels;
    }
}
