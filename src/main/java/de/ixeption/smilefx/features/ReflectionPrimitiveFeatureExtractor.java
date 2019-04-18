package de.ixeption.smilefx.features;

import com.google.common.collect.Lists;

import java.lang.reflect.Field;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;


public class ReflectionPrimitiveFeatureExtractor<T> implements FeatureExtractor<T, double[]> {

    public static final List<Class> SUPPORTED_CLASSES = Lists.newArrayList(boolean.class, int.class, long.class, double.class, float.class, byte.class, char.class,
            short.class, EnumSet.class);

    private final Class _clazz;
    private final List<FeatureType> _featureTypes;
    private final List<Field> _fields;
    private final List<String> _fieldNames;
    private final List<Integer> _ordinals;

    public ReflectionPrimitiveFeatureExtractor(Class clazz) {
        this._clazz = clazz;
        _featureTypes = new ArrayList<>();
        _fields = new ArrayList<>();
        _fieldNames = new ArrayList<>();
        _ordinals = new ArrayList<>();

        Field[] fields = clazz.getFields();
        for (Field field : fields) {
            if (field.getAnnotationsByType(IgnoreFeature.class).length == 0) {
                if (SUPPORTED_CLASSES.contains(field.getType())) {
                    if (field.getType().equals(boolean.class)) {
                        _featureTypes.add(FeatureType.Binary);
                        _fields.add(field);
                        _fieldNames.add(field.getName());
                        _ordinals.add(-1);
                    } else if (field.getType().equals(EnumSet.class)) {
                        Object[] objects = extractEnumConstants(field);
                        for (Object o : objects) {
                            _featureTypes.add(FeatureType.Binary);
                            _fields.add(field);
                            _fieldNames.add(field.getName() + ":" + o.toString());
                            _ordinals.add(((Enum) o).ordinal());
                        }
                    } else {
                        _featureTypes.add(FeatureType.Continuous);
                        _fields.add(field);
                        _fieldNames.add(field.getName());
                        _ordinals.add(-1);
                    }
                }
            }
        }

    }

    private static Object[] extractEnumConstants(Field field) {
        Type typeArgument = ((ParameterizedType) field.getGenericType()).getActualTypeArguments()[0];
        Object[] constants = ((Class) typeArgument).getEnumConstants();
        return constants;
    }

    @Override
    public double[] extract(T value) {
        double[] arr = new double[_fieldNames.size()];
        try {
            for (int i = 0; i < _fields.size(); i++) {
                Field f = _fields.get(i);
                if (f.getType().equals(boolean.class)) {
                    boolean o = (boolean) f.get(value);
                    arr[i] = o ? 1.0 : 0;
                } else if (f.getType().equals(EnumSet.class)) {
                    EnumSet<?> enumSet = (EnumSet<?>) f.get(value);
                    if (enumSet != null) {
                        int finalI = i;
                        if (enumSet.stream().anyMatch(o -> o.ordinal() == _ordinals.get(finalI))) {
                            arr[i] = 1.0;
                        }
                    }

                } else {
                    Number o = (Number) f.get(value);
                    arr[i] = o.doubleValue();
                }
            }
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        return arr;
    }

    @Override
    public String getFeatureNameForIndex(int index) {
        return _fieldNames.get(index);

    }

    @Override
    public String[] getFeatureNames() {
        return _fieldNames.toArray(new String[0]);
    }

    @Override
    public FeatureType[] getFeatureTypes() {
        return _featureTypes.toArray(new FeatureType[0]);

    }

    @Override
    public int getNumberOfFeatures() {
        return _fieldNames.size();
    }
}
