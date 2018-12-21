package de.ixeption.smilefx.util;

import java.io.*;
import java.nio.file.Path;


public final class PersistenceUtils {

    private PersistenceUtils() {
    }

    /**
     * @param fileName the file to save to
     * @param <T>      the object type
     * @return deserialized object
     * @throws IOException            if file is not valid
     * @throws ClassNotFoundException if the class is not found
     */
    @SuppressWarnings("unchecked")
    public static <T> T deserialize(Path fileName) throws IOException, ClassNotFoundException {
        T obj = null;
        try (FileInputStream fis = new FileInputStream(fileName.toFile());
             BufferedInputStream bis = new BufferedInputStream(fis);
             ObjectInputStream ois = new ObjectInputStream(bis)) {
            obj = (T) ois.readObject();
        }
        return obj;
    }

    /**
     * @param obj      the object to serialize
     * @param fileName the file to load from
     * @param <T>      the object type
     * @throws IOException if the file is not found
     */
    public static <T> void serialize(T obj, Path fileName) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(fileName.toFile());
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             ObjectOutputStream oos = new ObjectOutputStream(bos)) {
            oos.writeObject(obj);
        }
    }
}
