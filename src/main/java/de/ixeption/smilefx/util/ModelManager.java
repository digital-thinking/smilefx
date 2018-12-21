package de.ixeption.smilefx.util;


import de.ixeption.smilefx.training.PredictionWithThreshold;
import de.ixeption.smilefx.training.TrainedBinarySmileModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;


public class ModelManager {

    private static final String MODEL_FILE_EXTENSION = ".model";
    private static final String ML_MODEL_FOLDER = "ml_models";
    private static final String SEPERATOR = "_";
    private static final String DEFAULT_MODEL_PREFIX = "default";
    private static final Logger _log = LoggerFactory.getLogger(ModelManager.class);
    private LRUCache<String, TrainedBinarySmileModel> _smileModelLRUCache = new LRUCache<>(10);

    static int compareFileByTimestampPostfix(Path p1, Path p2) {
        final long s1 = Long.parseLong(p1.getFileName().toString().replace(MODEL_FILE_EXTENSION, "").split(SEPERATOR)[1]);
        final long s2 = Long.parseLong(p2.getFileName().toString().replace(MODEL_FILE_EXTENSION, "").split(SEPERATOR)[1]);
        return Long.compare(s1, s2);
    }

    public void clearCaches() {
        _smileModelLRUCache.clear();
    }

    public Set<String> getLoadedModels() {
        return _smileModelLRUCache.keySet();
    }

    public String saveModel(String identifier, TrainedBinarySmileModel trainedBinarySmileModel, long time) throws IOException {
        new File(ML_MODEL_FOLDER).mkdirs();
        String path = ML_MODEL_FOLDER + "/" + identifier + SEPERATOR + time + MODEL_FILE_EXTENSION;
        PersistenceUtils.serialize(trainedBinarySmileModel, Paths.get(path));
        return path;
    }

    public @Nullable
    PredictionWithThreshold getPrediction(String id, double[] features) {
        TrainedBinarySmileModel model = _smileModelLRUCache.get(id);
        if (model == null) {
            try {
                model = loadLastOrDefaultModel(id);
                if (model != null) {
                    _smileModelLRUCache.put(id, model);
                }
            } catch (IOException e) {
                _log.error("Did not find model: " + id, e);
            } catch (ClassNotFoundException e) {
                _log.error("Class for model not found: " + id, e);
            }
        }
        if (model != null && features != null) {
            return model.predict(features);
        }
        return null;
    }

    @Nullable
    TrainedBinarySmileModel loadLastOrDefaultModel(String identifier) throws IOException, ClassNotFoundException {
        Optional<Path> modelPath;
        try (Stream<Path> paths = Files.walk(Paths.get(ML_MODEL_FOLDER))) {
            modelPath = paths.filter(Files::isRegularFile)//
                    .filter(path -> path.getFileName().toString().endsWith(MODEL_FILE_EXTENSION))//
                    .filter(path -> !path.getFileName().toString().startsWith(DEFAULT_MODEL_PREFIX))//
                    .filter(path -> path.getFileName().toString().split(SEPERATOR)[0].equals(identifier))//
                    .max(ModelManager::compareFileByTimestampPostfix);
        }
        if (!modelPath.isPresent()) {
            try (Stream<Path> paths = Files.walk(Paths.get(ML_MODEL_FOLDER))) {
                modelPath = paths.filter(Files::isRegularFile)//
                        .filter(path -> path.getFileName().toString().endsWith(MODEL_FILE_EXTENSION))//
                        .filter(path -> path.getFileName().toString().startsWith(DEFAULT_MODEL_PREFIX))//
                        .filter(path -> path.getFileName().toString().split(SEPERATOR)[1].equals(identifier)).findFirst();
            }
        }
        if (modelPath.isPresent()) {
            return PersistenceUtils.deserialize(modelPath.get());
        }
        return null;
    }

    public static class LRUCache<K, V> extends LinkedHashMap<K, V> {

        private int cacheSize;


        public LRUCache(int cacheSize) {
            super(16, 0.75f, true);
            this.cacheSize = cacheSize;
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
            return size() >= cacheSize;
        }
    }
}
