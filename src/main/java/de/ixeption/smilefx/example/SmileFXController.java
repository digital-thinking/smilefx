package de.ixeption.smilefx.example;

import de.ixeption.smilefx.controller.AbstractController;
import de.ixeption.smilefx.features.FeatureExtractor;
import de.ixeption.smilefx.features.GenericFeatureExtractorBuilder;
import de.ixeption.smilefx.training.GridSearch;
import de.ixeption.smilefx.training.TrainingDataSet;
import javafx.scene.layout.Pane;
import org.jetbrains.annotations.NotNull;
import smile.classification.SVM;
import smile.math.kernel.LinearKernel;
import smile.stat.distribution.GaussianDistribution;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.function.Consumer;


public class SmileFXController extends AbstractController<double[], double[]> {

   @Override
   public Pane getDataInput() {
      final Pane pane = new Pane();
      return pane;

   }

   @Override
   public Class<double[]> getFeatureDataType() {
      return double[].class;
   }

   @Override
   public FeatureExtractor<double[], double[]> getFeatureExtractor() {
      final GenericFeatureExtractorBuilder<double[]> builder = new GenericFeatureExtractorBuilder<>();
      builder.addFeature("First", doubles -> doubles[0], FeatureExtractor.FeatureType.Binary);
      builder.addFeature("Second", doubles -> doubles[1], FeatureExtractor.FeatureType.Binary);
      builder.addFeature("Third", doubles -> doubles[2], FeatureExtractor.FeatureType.Continuous);
      return builder.build();
   }

   @Override
   public Pane getPredictInput() {
      return new Pane();
   }

   @NotNull
   @Override
   protected GridSearch<double[]> getGridSearch(EnumSet<GridSearch.MLModelType> modelTypes) {
      return new CustomGridSearch(modelTypes, Integer.parseInt(kFolds.getText()));
   }

   @Override
   protected String getModelIdentifier() {
      return "MyModel";
   }

   @Override
   protected TrainingDataSet<double[]> getTrainingData(double resampleRate, long limit, Consumer<Double> callback) {
      final TrainingDataSet<double[]> trainingDataSet = new TrainingDataSet<>(double[].class);
      GaussianDistribution normalDis = new GaussianDistribution(0.5, 1);
      trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{0, 1, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{1, 0, normalDis.rand()}, 1);
      trainingDataSet.addDatapoint(new double[]{0, 0, normalDis.rand()}, 0);
      trainingDataSet.addDatapoint(new double[]{1, 1, normalDis.rand()}, 0);
      return trainingDataSet;
   }

   @Override
   protected TrainingDataSet<double[]> getValidationData() throws Exception {
      final TrainingDataSet<double[]> trainingDataSet = new TrainingDataSet<>(double[].class);
      trainingDataSet.addDatapoint(new double[]{0, 1, 0.1}, 1);
      trainingDataSet.addDatapoint(new double[]{1, 1, 0.8}, 0);
      return trainingDataSet;
   }

   public static class CustomGridSearch extends GridSearch<double[]> {

      public CustomGridSearch(EnumSet<MLModelType> mLModelTypeToSearches, int foldk) {
         super(mLModelTypeToSearches, foldk, double[].class);
      }

      @Override
      protected void gridSearchModel(MLModelType model, double mean, int numberOfFeatures, Class<double[]> type) {
         switch (model) {
            case SVM_Linear:
               SVM.Trainer<double[]> trainer = new SVM.Trainer<>(new LinearKernel(), 0.3, 0.9);
               HashMap<String, String> params = new HashMap<>();
               params.put("CP", String.valueOf(0.3));
               params.put("CN", String.valueOf(0.9));
               addToCrossValidation(model, trainer, params);
         }

      }
   }
}
