package de.ixeption.smilefx.training;

import smile.classification.ClassifierTrainer;
import smile.classification.SVM;
import smile.classification.SoftClassifier;
import smile.validation.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;


public class SmileModelTrainer<T> implements Serializable {

    private static final long serialVersionUID = 3468475674256329188L;
    private final transient ClassifierTrainer<T> _classifierTrainer;


    public SmileModelTrainer(ClassifierTrainer<T> classifierTrainer) {
        _classifierTrainer = classifierTrainer;

    }

    public Map<String, Double> crossValidate(TrainingDataSet<T> trainingDataSet) {
        T[] array = trainingDataSet.getFeatures();
        int[] y = trainingDataSet.getLabels();
        return crossValidate(array, y);

    }

    public SoftClassifier<T> trainModel(T[] features, int[] labels) {
        final SoftClassifier<T> classifier = (SoftClassifier<T>) _classifierTrainer.train(features, labels);
        if (classifier instanceof SVM) {
            ((SVM<T>) classifier).trainPlattScaling(features, labels);
        }
        return classifier;
    }

    Map<String, Double> crossValidate(T[] array, int[] y) {
        double[] measures = Validation.cv(10, _classifierTrainer, array, y,
                new ClassificationMeasure[]{new Accuracy(), new Sensitivity(), new Precision(), new MCCMeasure()});
        HashMap<String, Double> map = new HashMap<>();
        map.put("Accuracy", measures[0]);
        map.put("Sensitivity", measures[1]);
        map.put("Precision", measures[2]);
        map.put("MatthewsCorrelationCoefficient", measures[3]);
        return map;
    }

}
