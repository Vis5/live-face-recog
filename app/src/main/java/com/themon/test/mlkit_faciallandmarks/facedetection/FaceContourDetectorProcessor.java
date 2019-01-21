package com.themon.test.mlkit_faciallandmarks.facedetection;

import android.graphics.Bitmap;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.themon.test.mlkit_faciallandmarks.VisionProcessorBase;
import com.themon.test.mlkit_faciallandmarks.common.CameraImageGraphic;
import com.themon.test.mlkit_faciallandmarks.common.FrameMetadata;
import com.themon.test.mlkit_faciallandmarks.common.GraphicOverlay;

import java.io.IOException;
import java.util.List;

/**
 * Face Contour Demo.
 */
public class FaceContourDetectorProcessor extends VisionProcessorBase<List<FirebaseVisionFace>> {

    private static final String TAG = "FaceContourDetectorProc";

    private final FirebaseVisionFaceDetector detector;
    //private PersonRecognizer mPersonRecog;

    public FaceContourDetectorProcessor(/*PersonRecognizer pr*/) {
        FirebaseVisionFaceDetectorOptions options =
                new FirebaseVisionFaceDetectorOptions.Builder()
                        .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
                        .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                        //.setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                        .build();

        detector = FirebaseVision.getInstance().getVisionFaceDetector(options);
        //mPersonRecog = pr;
    }

    @Override
    public void stop() {
        try {
            detector.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown while trying to close Face Contour Detector: " + e);
        }
    }

    @Override
    protected Task<List<FirebaseVisionFace>> detectInImage(FirebaseVisionImage image) {
        return detector.detectInImage(image);
    }

    @Override
    protected void onSuccess(
            @Nullable Bitmap originalCameraImage,
            @NonNull List<FirebaseVisionFace> faces,
            @NonNull FrameMetadata frameMetadata,
            @NonNull GraphicOverlay graphicOverlay) {
        this.faces = faces;
        //Mat m = new Mat();
        graphicOverlay.clear();
        if (originalCameraImage != null) {
            CameraImageGraphic imageGraphic = new CameraImageGraphic(graphicOverlay, originalCameraImage);
            graphicOverlay.add(imageGraphic);
            //Utils.bitmapToMat(originalCameraImage, m);
        }
        for (int i = 0; i < faces.size(); ++i) {
            FirebaseVisionFace face = faces.get(i);
            /*Rect bbox = face.getBoundingBox();
            Mat subm = new Mat(m, new org.opencv.core.Rect(bbox.left, bbox.top, bbox.width(), bbox.height()));
            String label = mPersonRecog.predict(subm);
            */
            FaceContourGraphic faceGraphic = new FaceContourGraphic(graphicOverlay, face);
            graphicOverlay.add(faceGraphic);
        }
        graphicOverlay.postInvalidate();
        //m.release();
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }

}