package de.ixeption.smilefx.example;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Modality;
import javafx.stage.Stage;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;


public class SmileFXApp extends Application {

    private static void showError(Thread t, Throwable e) {
        if (Platform.isFxApplicationThread()) {
            showErrorDialog(e);
        } else {
            Platform.runLater(() -> showErrorDialog(e));
        }
    }

    private static void showErrorDialog(Throwable e) {
        StringWriter errorMsg = new StringWriter();
        e.printStackTrace(new PrintWriter(errorMsg));
        Stage dialog = new Stage();
        dialog.initModality(Modality.APPLICATION_MODAL);
        final FXMLLoader fxmlLoader = new FXMLLoader(SmileFXApp.class.getResource("/error.fxml"));
        try {
            Parent root = fxmlLoader.load();
            ((ErrorController) fxmlLoader.getController()).setErrorText(errorMsg.toString());
            dialog.setScene(new Scene(root, 600, 400));
            dialog.show();

        } catch (IOException exc) {
            exc.printStackTrace();
        }
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        Thread.setDefaultUncaughtExceptionHandler(SmileFXApp::showError);
        final FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/ml-tool.fxml"));
        final SmileFXController smileFXController = new SmileFXController();
        smileFXController.setStage(primaryStage);
        final Parent root = fxmlLoader.load();
        primaryStage.setTitle("SmileFx Application");
        primaryStage.setScene(new Scene(root, 1280, 768));
        primaryStage.getIcons().add(new Image(getClass().getResource("/icon.png").openStream()));
        primaryStage.setResizable(true);
        primaryStage.show();

        primaryStage.setOnCloseRequest(event -> Platform.exit());
    }

}
