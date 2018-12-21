package de.ixeption.smilefx.training;

import de.ixeption.smilefx.features.FeatureExtractor;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.classification.AdaBoost;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;
import smile.classification.*;
import smile.feature.Scaler;
import smile.feature.SignalNoiseRatio;
import smile.feature.SumSquaresRatio;
import smile.math.SparseArray;
import smile.math.kernel.*;
import smile.validation.ClassificationMeasure;

import javax.annotation.Nullable;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static de.ixeption.smilefx.training.GridSearch.MLModelType.*;
import static java.util.stream.Collectors.toMap;


public class GridSearch<T> {

    public static final int MAX_ITERATION = 100000;
    private static final Logger _log = LoggerFactory.getLogger(GridSearch.class);
    private final EnumSet<MLModelType> _models;
    private final int _foldk;
    private final Class<T> _type;
    private List<GridSearchResult> _gridSearchResults;
    private List<GridSearchResult> _results;


    /**
     * @param MLModelTypeToSearches Set of Models to be used in grid search    *
     * @param foldk                 number of folds for cross-validation, 10 is a good default
     */
    public GridSearch(EnumSet<MLModelType> MLModelTypeToSearches, int foldk, Class<T> type) {
        this._models = MLModelTypeToSearches;
        this._foldk = foldk;
        this._type = type;
        _gridSearchResults = new ArrayList<>();
    }

    /**
     * Uses grid search and cross validation to find the best model
     * uses multi-threading
     *
     * @param features             the features to train the model (may not be scaled)
     * @param labels               the labels
     * @param measures             the measures to execute
     * @param measureForComparsion the measure to use for comparsion (simple class name)
     * @param featureExtractor     the feature extractor
     * @return the best model or null
     */
    public @Nullable
    TrainedBinarySmileModel<T> findBestModel(T[] features, int[] labels, ClassificationMeasure[] measures, String measureForComparsion,
                                             FeatureExtractor<?, T> featureExtractor) {
        T[] x;
        Scaler scaler = null;
        if (features instanceof double[][]) {
            scaler = new Scaler();
            scaler.learn((double[][]) features);
            x = (T[]) scaler.transform((double[][]) features);
            printFeatureImportance(labels, (double[][]) x, 20, featureExtractor);
        } else {
            x = features;
        }

        final long before = System.currentTimeMillis();
        _log.info("Starting grid search for {}" + _models);
        _results = gridSearch(_models, x, labels, featureExtractor.getNumberOfFeatures(), measures,
                Runtime.getRuntime().availableProcessors() > 2 ? Runtime.getRuntime().availableProcessors() - 1 : 1, null);
        _log.info("Finished grid search in {} ms", System.currentTimeMillis() - before);

        final Optional<GridSearchResult> best = _results.stream().max(Comparator.comparingDouble(gs -> {
            final double measure = gs.getcVresult().getMeasure(measureForComparsion);
            return Double.isNaN(measure) ? 0.0 : measure;
        }));
        if (best.isPresent()) {
            _log.info("-------------------------------------------");
            GridSearch.GridSearchResult result = best.get();

            _log.info("Found best model {}: {} @ {}", best.get().getMLModelType(), result.getcVresult().getMeasure(measureForComparsion),
                    Arrays.toString(result.getParams().entrySet().toArray()));
            SmileModelTrainer<T> smileModelTrainer = new SmileModelTrainer<>(result.getClassifierTrainer());
            final TrainedBinarySmileModel<T> smileModel = new TrainedBinarySmileModel<>(smileModelTrainer.trainModel(x, labels), scaler, null, 0.5);
            if (smileModel.getImportancesIfAvailable().isPresent()) {
                double[] importances = smileModel.getImportancesIfAvailable().get();
                String[] labelNames = featureExtractor.getFeatureNames();
                _log.info("Importances (>0): ");
                for (int i = 0; i < importances.length; i++) {
                    if (importances[i] > 0) {
                        _log.info(labelNames[i] + ": " + importances[i]);
                    }
                }
            }

            return smileModel;
        }
        return null;

    }

    @SuppressWarnings("unchecked")
    public List<GridSearchResult> gridSearch(EnumSet<MLModelType> models, T[] data, int[] labels, int numFeatures, ClassificationMeasure[] measures,
                                             int parallelism, Consumer<Double> progressCallback) {
        final double mean = Arrays.stream(labels).average().orElseThrow(IllegalArgumentException::new);

        _log.info("Dataset size: {} number of features: {} mean label: {}", labels.length, numFeatures, mean);
        models.forEach(m -> gridSearchModel(m, mean, numFeatures, _type));
        Map<ClassifierTrainer<T>, BalancedCrossValidation.CVresult> map = BalancedCrossValidation.bcv(_foldk, data, labels, measures, parallelism,
                progressCallback, false, _gridSearchResults.stream().map(gridSearchResult -> gridSearchResult._classifierTrainer)//
                        .toArray(ClassifierTrainer[]::new));//

        for (GridSearchResult res : _gridSearchResults) {
            res.setcVresult(map.get(res.getClassifierTrainer()));
        }
        return _gridSearchResults;

    }

    public void gridSearchModel(MLModelType model, double mean, int numberOfFeatures, Class<T> type) {
        final int[] treeSizes = {10, 100, 200};
        final int[] nodeSizes = {5, 10, 20};

        List<Pair<Double, Double>> cs = new ArrayList<>();
        cs.add(Pair.of(0.4, 0.4));
        cs.add(Pair.of(1.0, 1.0));
        cs.add(Pair.of(0.4, 0.4 * mean));
        cs.add(Pair.of(1.0, 1.0 * mean / 2));
        cs.add(Pair.of(5.0, 5.0 * mean / 5));

        if (type.equals(SparseArray.class)) {
            switch (model) {
                case SVM_Linear:
                    gridSearchSparseSVM(SVM_Linear, cs, new SparseLinearKernel());
                    break;
                case SVM_Gaussian:
                    gridSearchSparseSVM(SVM_Gaussian, cs, //
                            new SparseGaussianKernel(0.5), //
                            new SparseGaussianKernel(1), //
                            new SparseGaussianKernel(5));
                    break;
                case SVM_Laplacian:
                    gridSearchSparseSVM(SVM_Laplacian, cs, //
                            new SparseLaplacianKernel(0.5), //
                            new SparseLaplacianKernel(1), //
                            new SparseLaplacianKernel(5));
                    break;
            }
        } else {
            switch (model) {
                case SVM_Linear:
                    gridSearchSVM(SVM_Linear, cs, new LinearKernel());
                    break;
                case SVM_Gaussian:
                    gridSearchSVM(SVM_Gaussian, cs, //
                            new GaussianKernel(0.5), //
                            new GaussianKernel(1), //
                            new GaussianKernel(5));
                    break;
                case SVM_Laplacian:
                    gridSearchSVM(SVM_Laplacian, cs, //
                            new LaplacianKernel(0.5), //
                            new LaplacianKernel(1), //
                            new LaplacianKernel(5));
                    break;
                case AdaBoost:
                    gridSearchAdaBoost(treeSizes, nodeSizes);
                    break;
                case RandomForest:
                    gridSearchRandomForest(treeSizes, nodeSizes, numberOfFeatures);
                    break;
                case GradientBoostedTree:
                    gridSearchGradientBoostedTree(treeSizes, nodeSizes, new double[]{0.001, 0.01, 0.05});
                    break;
                case LogisticRegression:
                    gridSearchLogisticRegression(new int[]{1, 2, 10});
                    break;
            }
        }

    }

    private void addToCrossValidation(MLModelType mlModelType, ClassifierTrainer<T> trainer, HashMap<String, String> params) {
        _gridSearchResults.add(new GridSearchResult(params, trainer, mlModelType));
    }

    private void gridSearchAdaBoost(int[] treeSizes, int[] nodeSizes) {
        for (int treeSize : treeSizes) {
            for (int nodeSize : nodeSizes) {
                AdaBoost.Trainer trainer = new AdaBoost.Trainer(treeSize);
                trainer.setMaxNodes(nodeSize);
                HashMap<String, String> params = new HashMap<>();
                params.put("nodeSize", String.valueOf(nodeSize));
                params.put("treeSize", String.valueOf(treeSize));
                addToCrossValidation(MLModelType.AdaBoost, (ClassifierTrainer<T>) trainer, params);
            }
        }
    }

    private void gridSearchGradientBoostedTree(int[] treeSizes, int[] nodeSizes, double[] shrinkages) {
        for (int treeSize : treeSizes) {
            for (int nodeSize : nodeSizes) {
                for (double shrinkage : shrinkages) {
                    GradientTreeBoost.Trainer trainer = new GradientTreeBoost.Trainer(treeSize);
                    trainer.setMaxNodes(nodeSize);
                    trainer.setShrinkage(shrinkage);
                    HashMap<String, String> params = new HashMap<>();
                    params.put("treeSize", String.valueOf(treeSize));
                    params.put("nodeSize", String.valueOf(nodeSize));
                    params.put("shrinkage", String.valueOf(shrinkage));
                    addToCrossValidation(MLModelType.GradientBoostedTree, (ClassifierTrainer<T>) trainer, params);
                }
            }
        }
    }

    private void gridSearchLogisticRegression(int[] lambdas) {
        for (int lamda : lambdas) {
            LogisticRegression.Trainer trainer = new LogisticRegression.Trainer();
            trainer.setRegularizationFactor(lamda);
            trainer.setMaxNumIteration(MAX_ITERATION);
            HashMap<String, String> params = new HashMap<>();
            params.put("lamda", String.valueOf(lamda));
            addToCrossValidation(MLModelType.LogisticRegression, (ClassifierTrainer<T>) trainer, params);

        }
    }

    private void gridSearchRandomForest(int[] treeSizes, int[] nodeSizes, int numberOfFeatures) {
        for (int treeSize : treeSizes) {
            for (int nodeSize : nodeSizes) {
                RandomForest.Trainer trainer = new RandomForest.Trainer(treeSize);
                trainer.setNodeSize(nodeSize);
                trainer.setSplitRule(DecisionTree.SplitRule.GINI);
                trainer.setNumRandomFeatures((int) Math.sqrt(numberOfFeatures));
                HashMap<String, String> params = new HashMap<>();
                params.put("treeSize", String.valueOf(treeSize));
                params.put("nodeSize", String.valueOf(nodeSize));
                addToCrossValidation(MLModelType.RandomForest, (ClassifierTrainer<T>) trainer, params);

            }
        }
    }

    @SafeVarargs
    private final void gridSearchSVM(MLModelType mlModelType, List<Pair<Double, Double>> cs, MercerKernel<double[]>... kernels) {
        for (MercerKernel<double[]> mercerKernel : kernels) {
            for (Pair<Double, Double> c : cs) {
                SVM.Trainer<double[]> trainer = new SVM.Trainer<>(mercerKernel, c.getLeft(), c.getRight());
                trainer.setMaxIter(MAX_ITERATION);
                HashMap<String, String> params = new HashMap<>();
                params.put("CP", String.valueOf(c.getLeft()));
                params.put("CN", String.valueOf(c.getRight()));
                addToCrossValidation(mlModelType, (ClassifierTrainer<T>) trainer, params);

            }
        }

    }

    private void gridSearchSparseSVM(MLModelType mlModelType, List<Pair<Double, Double>> cs, MercerKernel<SparseArray>... kernels) {
        for (MercerKernel<SparseArray> mercerKernel : kernels) {
            for (Pair<Double, Double> c : cs) {
                SVM.Trainer<SparseArray> trainer = new SVM.Trainer<>(mercerKernel, c.getLeft(), c.getRight());
                trainer.setMaxIter(MAX_ITERATION);
                HashMap<String, String> params = new HashMap<>();
                params.put("CP", String.valueOf(c.getLeft()));
                params.put("CN", String.valueOf(c.getRight()));
                addToCrossValidation(mlModelType, (ClassifierTrainer<T>) trainer, params);

            }
        }
    }

    private void printFeatureImportance(int[] labels, double[][] x, int topn, FeatureExtractor<?, T> featureExtractor) {
        final int classes = IntStream.of(labels).distinct().toArray().length;
        final double[] rank;
        if (classes == 2) {
            _log.info("Binary classification: feature importance (SNR)");
            SignalNoiseRatio signalNoiseRatio = new SignalNoiseRatio();
            rank = signalNoiseRatio.rank(x, labels);

        } else {
            _log.info("Multi-class classification: eature importance (SNR)");
            SumSquaresRatio sumSquaresRatio = new SumSquaresRatio();
            rank = sumSquaresRatio.rank(x, labels);

        }
        IntStream.range(0, rank.length).boxed()//
                .collect(toMap(featureExtractor::getFeatureNameForIndex, i -> rank[i]))//
                .entrySet().stream()//
                .filter(e -> !e.getValue().isNaN()).sorted((o1, o2) -> Double.compare(Math.abs(o2.getValue()), Math.abs(o1.getValue()))).limit(topn)
                .forEach(e -> _log.info("{}\t\t{}", e.getKey(), e.getValue()));
    }


    // @formatter:off
    public enum MLModelType {
        SVM_Linear, SVM_Gaussian, SVM_Laplacian,
        RandomForest, AdaBoost, GradientBoostedTree,
        NaiveBayes, LogisticRegression

    }
    // @formatter:on

    public static class GridSearchResult {

        private final Map<String, String> params;
        private final ClassifierTrainer _classifierTrainer;
        private BalancedCrossValidation.CVresult _cVresult;
        private MLModelType _mlModelType;


        public GridSearchResult(Map<String, String> params, ClassifierTrainer classifierTrainer, MLModelType mlModelType) {
            this.params = params;
            _classifierTrainer = classifierTrainer;
            _mlModelType = mlModelType;
        }

        public ClassifierTrainer getClassifierTrainer() {
            return _classifierTrainer;
        }

        public double getF1() {
            return _cVresult._map.get("FMeasure");
        }

        public MLModelType getMLModelType() {
            return _mlModelType;
        }

        public double getMcc() {
            return _cVresult._map.get("MCCMeasure");
        }

        public Map<String, String> getParams() {
            return params;
        }

        public BalancedCrossValidation.CVresult getcVresult() {
            return _cVresult;
        }

        public void setcVresult(BalancedCrossValidation.CVresult cVresult) {
            _cVresult = cVresult;
        }
    }

}
