package de.ixeption.smilefx.example;

import javafx.application.Application;


public class SmileFxRunner {


    public static void main(String[] args) {
        new SmileFxRunner().run();
    }

    protected void run() {
        Application.launch(SmileFXApp.class);
    }


}
