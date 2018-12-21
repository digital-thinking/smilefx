package de.ixeption.smilefx;

import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.training.GridSearch;
import de.ixeption.smilefx.training.TrainedBinarySmileModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import smile.classification.SoftClassifier;
import smile.validation.Accuracy;
import smile.validation.ClassificationMeasure;
import smile.validation.MCCMeasure;

import java.util.EnumSet;

import static org.assertj.core.api.Assertions.assertThat;


public class GridSearchTest {

    private double[][] _x;
    private int[] _y;


    @BeforeEach
    public void generateTrainingData() {
        _x = new double[][]{{0, 1, 0, 1}, //
                {0, 0, 1, 1}, //
                {0, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 0, 1, 1}, //
                {1, 0, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {0, 1, 0, 1}, //
                {0, 0, 1, 1}, //
                {0, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 0, 1, 1}, //
                {1, 0, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 1, 1, 1}, //
                {1, 1, 1, 1}, //
        };
        _y = new int[]{0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1};
    }

    @Test
    public void testFindBestModel() {

        final FeatureExtractor<double[], double[]> featureExtractor = new FeatureExtractor<double[], double[]>() {

            @Override
            public double[] extract(double[] value) {
                return value;
            }

            @Override
            public String getFeatureNameForIndex(int index) {
                return "Label " + index;
            }

            @Override
            public String[] getFeatureNames() {
                return new String[]{"1", "2", "3", "4"};
            }

            @Override
            public FeatureType[] getFeatureTypes() {
                return new FeatureType[]{FeatureType.Binary, FeatureType.Binary, FeatureType.Binary, FeatureType.Binary};
            }

            @Override
            public int getNumberOfFeatures() {
                return 4;
            }
        };

        GridSearch<double[]> gridSearch = new GridSearch<>(EnumSet.allOf(GridSearch.MLModelType.class), 3, double[].class);
        TrainedBinarySmileModel<double[]> bestModel = gridSearch.findBestModel(_x, _y, new ClassificationMeasure[]{new Accuracy(), new MCCMeasure()},
                "MCCMeasure", featureExtractor);
        assertThat(bestModel.getClassifier()).isInstanceOf(SoftClassifier.class);

    }
}