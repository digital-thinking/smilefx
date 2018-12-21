package de.ixeption.smilefx.util;

import de.ixeption.smilefx.features.FeatureExtractor;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.util.Arrays;


public class TablesawConverter {

    public static <T, R> Table toTable(String name, T[] data, FeatureExtractor<T, R> featureExtractor) {
        String[] featureNames = featureExtractor.getFeatureNames();
        FeatureExtractor.FeatureType[] featureTypes = featureExtractor.getFeatureTypes();
        double[][] rowlike = Arrays.stream(data).map(featureExtractor::extract).toArray(double[][]::new);
        return toTable(name, featureNames, featureTypes, rowlike);
    }

    public static <T> Table toTable(String tableName, String[] featureNames, FeatureExtractor.FeatureType[] featureTypes, T[] rowlike) {
        if (rowlike instanceof double[][]) {
            double[][] array = (double[][]) rowlike;
            final Table table = Table.create(tableName);
            double[][] columnlike = new double[array[0].length][array.length];
            for (int x = 0; x < array.length; x++) {
                for (int y = 0; y < array[0].length; y++) {
                    columnlike[y][x] = array[x][y];
                }
            }

            for (int i = 0; i < featureNames.length; i++) {
                if (featureTypes[i].equals(FeatureExtractor.FeatureType.Binary)) {
                    table.addColumns(BooleanColumn.create(featureNames[i], Arrays.stream(columnlike[i]).mapToObj(d -> d > 0.5).toArray(Boolean[]::new)));
                } else {
                    table.addColumns(DoubleColumn.create(featureNames[i], columnlike[i]));
                }
            }
            return table;
        } else
            return null;

    }

}
