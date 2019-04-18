package de.ixeption.smilefx.training;

import de.ixeption.smilefx.util.PrecisionRecallCurve;
import de.ixeption.smilefx.util.RocCurve;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.classification.Classifier;
import smile.classification.ClassifierTrainer;
import smile.classification.SVM;
import smile.classification.SoftClassifier;
import smile.math.Math;
import smile.validation.ClassificationMeasure;
import smile.validation.ConfusionMatrix;
import smile.validation.CrossValidation;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;


public class BalancedCrossValidation {

    private static final Logger _log = LoggerFactory.getLogger(BalancedCrossValidation.class);
    private static ForkJoinPool THREAD_POOL;

    /**
     * balanced Cross validation of a classification model.
     *
     * @param k                k-fold cross validation.
     * @param trainers         classifiers trainer that are properly parameterized.
     * @param x                the test data set.
     * @param y                the test data labels.
     * @param parallelism      threads for training
     * @param measures         the performance measures of classification.    *
     * @param progressCallback Callback for progress notification, can be null and is called from ANY thread
     * @return {@link CVResult} the test results as a Map
     */

    public static <T> Map<ClassifierTrainer<T>, CVResult> bcv(int k, T[] x, int[] y, ClassificationMeasure[] measures, int parallelism,
                                                              @Nullable Consumer<Double> progressCallback, boolean resampling, ClassifierTrainer<T>... trainers) {
        if (k < 2) {
            throw new IllegalArgumentException("Invalid k for k-fold cross validation: " + k);
        }
        if (parallelism < 1) {
            throw new IllegalArgumentException("Invalid parallelism: " + parallelism);
        }
        if (Arrays.stream(y).distinct().count() > 2) {
            throw new IllegalArgumentException("Invalid class numbers: " + Arrays.stream(y).distinct().count());
        }

        _log.info("{}-fold Cross validation", k);
        Map<ClassifierTrainer<T>, CVResult> resultMap = new HashMap<>();

        int n = x.length;
        double[][] predictions = new double[trainers.length][n];
        CrossValidation cv = new CrossValidation(n, k);
        AtomicInteger progress = new AtomicInteger();
        for (int i = 0; i < k; i++) {
            T[] trainx;
            int[] trainy;
            if (resampling) {
                Resampler<?> resampler = new Resampler<>(Math.slice(x, cv.train[i]), Math.slice(y, cv.train[i]));
                trainx = (T[]) resampler.getFeatures();
                trainy = resampler.getLabels().stream().mapToInt(value -> value).toArray();
            } else {
                trainx = Math.slice(x, cv.train[i]);
                trainy = Math.slice(y, cv.train[i]);
            }

            THREAD_POOL = new ForkJoinPool(parallelism);
            for (int j = 0; j < trainers.length; j++) {
                ClassifierTrainer<T> t = trainers[j];
                int finalI = i;
                int finalJ = j;
                THREAD_POOL.execute(() -> {
                    Classifier<T> classifier = t.train(trainx, trainy);
                    if (classifier instanceof SVM) {
                        ((SVM<T>) classifier).trainPlattScaling(trainx, trainy);
                    }

                    final String simpleName = t.getClass().getDeclaringClass() == null ? "?" : t.getClass().getDeclaringClass().getSimpleName();
                    _log.debug("Training finished k:{}/n:{} - {} {}/{}", finalI + 1, k, simpleName, finalJ + 1, trainers.length);
                    if (progressCallback != null) {
                        progressCallback.accept(progress.incrementAndGet() / (double) (k * trainers.length));
                    }
                    for (int v : cv.test[finalI]) {
                        if (classifier instanceof SoftClassifier) {
                            double[] proba = new double[2];
                            final int predict = ((SoftClassifier<T>) classifier).predict(x[v], proba);
                            predictions[finalJ][v] = proba[1];
                        } else {
                            predictions[finalJ][v] = classifier.predict(x[v]);
                        }
                    }
                });

            }
            try {
                THREAD_POOL.shutdown();
                THREAD_POOL.awaitTermination(1, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                _log.info("Error", e);
            }
        }

        for (int i = 0; i < trainers.length; i++) {
            final int[] predictionClasses = Arrays.stream(predictions[i]).mapToInt(d -> (d == Math.floor(d)) && !Double.isInfinite(d) ? (int) d : d > 0.5 ? 1 : 0)
                    .toArray();
            ClassifierTrainer<T> t = trainers[i];
            final ConfusionMatrix matrix = new ConfusionMatrix(y, predictionClasses);
            final RocCurve roc = new RocCurve(y, predictions[i]);
            final PrecisionRecallCurve prc = new PrecisionRecallCurve(y, predictions[i]);

            CVResult cVresult = new CVResult(matrix, roc, prc);
            for (ClassificationMeasure measure : measures) {
                cVresult.addMeasure(measure.getClass().getSimpleName(), measure.measure(y, predictionClasses));
            }
            resultMap.put(t, cVresult);
        }
        return resultMap;
    }

    public static void stopAll() {
        if (THREAD_POOL != null) {
            THREAD_POOL.shutdownNow();
        }
    }

}
