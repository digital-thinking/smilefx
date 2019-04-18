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

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;


public class BalancedTrainTestSplitValidation {

    private static final Logger _log = LoggerFactory.getLogger(BalancedTrainTestSplitValidation.class);


    /**
     * @param trainX           training data
     * @param trainY           training labels
     * @param testX            test data
     * @param testY            test labels
     * @param measures         measures for validation
     * @param parallelism      threads for training
     * @param progressCallback Callback for progress notification, can be null and is called from ANY thread
     * @param resampling       resampling
     * @param trainers         classifiers trainer that are properly parameterized.
     * @return {@link CVResult} the test results as a Map
     */
    public static <T> Map<ClassifierTrainer<T>, CVResult> bttsv(T[] trainX, int[] trainY, T[] testX, int[] testY, ClassificationMeasure[] measures,
                                                                int parallelism, @Nullable Consumer<Double> progressCallback, boolean resampling, ClassifierTrainer<T>... trainers) {
        if (parallelism < 1) {
            throw new IllegalArgumentException("Invalid parallelism: " + parallelism);
        }
        if (Arrays.stream(trainY).distinct().count() > 2) {
            throw new IllegalArgumentException("Invalid class numbers: " + Arrays.stream(trainY).distinct().count());
        }

        Map<ClassifierTrainer<T>, CVResult> resultMap = new HashMap<>();

        double[][] predictions = new double[trainers.length][testY.length];
        AtomicInteger progress = new AtomicInteger();
        if (resampling) {
            Resampler<?> resampler = new Resampler<>(trainX, trainY);
            trainX = (T[]) resampler.getFeatures();
            trainY = resampler.getLabels().stream().mapToInt(value -> value).toArray();
        }

        ForkJoinPool forkJoinPool = new ForkJoinPool(parallelism);
        for (int j = 0; j < trainers.length; j++) {
            ClassifierTrainer<T> t = trainers[j];
            int finalJ = j;
            T[] finalTrainX = trainX;
            int[] finalTrainY = trainY;
            forkJoinPool.execute(() -> {
                Classifier<T> classifier = t.train(finalTrainX, finalTrainY);
                if (classifier instanceof SVM) {
                    ((SVM<T>) classifier).trainPlattScaling(finalTrainX, finalTrainY);
                }

                final String simpleName = t.getClass().getDeclaringClass() == null ? "?" : t.getClass().getDeclaringClass().getSimpleName();
                _log.debug("Training finished {} {}/{}", simpleName, finalJ + 1, trainers.length);
                if (progressCallback != null) {
                    progressCallback.accept(progress.incrementAndGet() / (double) (trainers.length));
                }
                for (int i = 0; i < testY.length; i++) {
                    if (classifier instanceof SoftClassifier) {
                        double[] proba = new double[2];
                        final int predict = ((SoftClassifier<T>) classifier).predict(testX[i], proba);
                        predictions[finalJ][i] = proba[1];
                    } else {
                        predictions[finalJ][i] = classifier.predict(testX[i]);
                    }
                }
            });

        }
        try {
            forkJoinPool.shutdown();
            forkJoinPool.awaitTermination(1, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            _log.info("Error", e);
        }

        for (int i = 0; i < trainers.length; i++) {
            final int[] predictionClasses = Arrays.stream(predictions[i]).mapToInt(d -> (d == Math.floor(d)) && !Double.isInfinite(d) ? (int) d : d > 0.5 ? 1 : 0)
                    .toArray();
            ClassifierTrainer<T> t = trainers[i];
            final ConfusionMatrix matrix = new ConfusionMatrix(testY, predictionClasses);
            final RocCurve roc = new RocCurve(testY, predictions[i]);
            final PrecisionRecallCurve prc = new PrecisionRecallCurve(testY, predictions[i]);

            CVResult cVresult = new CVResult(matrix, roc, prc);
            for (ClassificationMeasure measure : measures) {
                cVresult.addMeasure(measure.getClass().getSimpleName(), measure.measure(testY, predictionClasses));
            }
            resultMap.put(t, cVresult);
        }
        return resultMap;
    }

}
