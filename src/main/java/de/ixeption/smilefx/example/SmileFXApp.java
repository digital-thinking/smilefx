package de.ixeption.smilefx.example;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;


public class SmileFXApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        final FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/ml-tool.fxml"));
        final SmileFXController smileFXController = new SmileFXController();
        smileFXController.setStage(primaryStage);
        final Parent root = fxmlLoader.load();
        primaryStage.setTitle("Example Application");
        primaryStage.setScene(new Scene(root, 1280, 768));
        primaryStage.getIcons().add(new Image(getClass().getResource("/icon.png").openStream()));
        primaryStage.setResizable(true);
        primaryStage.show();

        primaryStage.setOnCloseRequest(event -> Platform.exit());
    }
}
