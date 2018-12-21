# Text Classify
[![Build Status](https://travis-ci.org/digital-thinking/text-classification.svg?branch=master)](https://travis-ci.org/digital-thinking/text-classification)

Out of the box text classification library in java built on top of the [smile framework](https://github.com/haifengl/smile).

## Sentiment analysis

```java
      // https://www.kaggle.com/c/si650winter11/data
      TrainingData data = new TrainingData("data/training.txt");
      ArrayList<TextFeature> textFeatures = data.getTextFeatures();
      ArrayList<Integer> labels = data.getLabels();
      FeatureUtils.shuffle(textFeatures, labels);

      NormalizingTextTokenizer normalizingTextTokenizer = new NormalizingTextTokenizer();
      StemmingNGrammProcessor filteringProcessor = new StemmingNGrammProcessor(3, 3, 25);

      Set<Token> corpus = textFeatures.stream().map(normalizingTextTokenizer::tokenize).map(filteringProcessor::process).flatMap(Arrays::stream).collect(Collectors.toSet());
      DefaultTextPipeline pipeline = new DefaultTextPipeline(new BagOfWordsFeatureExtractor(corpus), normalizingTextTokenizer);
      pipeline.getTokenProcessors().add(filteringProcessor);

      BinaryTextClassifierTrainer binaryTextClassifierTrainer = new BinaryTextClassifierTrainer(0.5, 0.5, pipeline);

      // cross validate
      BinaryUtils.BinaryConfusionMatrixMeasure confusionMatrixMeasure = binaryTextClassifierTrainer
              .crossValidate(textFeatures.toArray(new TextFeature[0]), labels.stream().mapToInt(Integer::intValue).toArray());
      log.info(" confusion matrix\n" + confusionMatrixMeasure);

      // train and validate
      int trainSize = 6000;
      TextFeature[] trainX = textFeatures.stream().limit(trainSize).toArray(TextFeature[]::new);
      int[] trainY = labels.stream().limit(trainSize).mapToInt(Integer::intValue).toArray();

      TextFeature[] testX = textFeatures.stream().skip(trainSize).toArray(TextFeature[]::new);
      int[] testY = labels.stream().skip(trainSize).mapToInt(Integer::intValue).toArray();

      TextClassifier classifier = binaryTextClassifierTrainer.train(trainX, trainY, null);
      int[] predictions = Arrays.stream(testX).map(classifier::predict).mapToInt(Prediction::getLabel).toArray();

      int correct = 0;
      for (int i = 0; i < testY.length; i++) {
         if (predictions[i] == testY[i]) {
            correct++;
         }

      }

      double accurancy = correct / (double) testY.length;
      log.info("Accuracy: " + accurancy);
                            
   
```
