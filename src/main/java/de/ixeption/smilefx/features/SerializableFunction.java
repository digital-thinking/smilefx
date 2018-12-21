package de.ixeption.smilefx.features;

import java.io.Serializable;
import java.util.function.Function;


@FunctionalInterface
public interface SerializableFunction<I, O> extends Function<I, O>, Serializable {
}