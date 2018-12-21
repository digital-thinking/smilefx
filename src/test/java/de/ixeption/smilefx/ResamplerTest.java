package de.ixeption.smilefx;


import de.ixeption.smilefx.training.Resampler;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class ResamplerTest {

    @Test
    public void testAutoResampling() {
        Resampler<Integer> cut = new Resampler<>(new Integer[]{0, 1, 2, 3}, new int[]{0, 0, 0, 1});
        assertThat(cut.getFeatures().length).isEqualTo(6);
    }

    @Test
    public void testResampling() {
        Resampler<Integer> cut = new Resampler<>(new Integer[]{0, 1, 2, 3}, new int[]{0, 0, 0, 1}, 1.0, 2.0);
        assertThat(cut.getFeatures().length).isEqualTo(5);
    }

}