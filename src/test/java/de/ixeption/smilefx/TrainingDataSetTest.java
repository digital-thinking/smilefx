package de.ixeption.smilefx;

import de.ixeption.smilefx.training.TrainingDataSet;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;


public class TrainingDataSetTest {

    @Test
    public void testAdd() {
        TrainingDataSet cut = new TrainingDataSet(double[].class);
        cut.addDatapoint(new double[]{1}, 1);
        cut.addDatapoint(new double[]{1}, 1);
        cut.addDatapoint(new double[]{1}, 1);
        cut.addDatapoint(new double[]{1}, 0);
        assertThat(cut.getSize()).isEqualTo(4);
    }

}