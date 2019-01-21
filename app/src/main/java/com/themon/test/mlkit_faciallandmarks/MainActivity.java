package com.themon.test.mlkit_faciallandmarks;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.text.InputType;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.themon.test.mlkit_faciallandmarks.common.BitmapUtils;
import com.themon.test.mlkit_faciallandmarks.common.CameraSource;
import com.themon.test.mlkit_faciallandmarks.common.CameraSourcePreview;
import com.themon.test.mlkit_faciallandmarks.common.FrameMetadata;
import com.themon.test.mlkit_faciallandmarks.common.GraphicOverlay;
import com.themon.test.mlkit_faciallandmarks.facedetection.FaceContourDetectorProcessor;
import com.themon.test.mlkit_faciallandmarks.facedetection.FaceDetectionProcessor;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceRec;
import com.tzutalin.dlib.FileUtils;
import com.tzutalin.dlib.VisionDetRet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback,
        AdapterView.OnItemSelectedListener,
        CompoundButton.OnCheckedChangeListener {
    private static final String FACE_DETECTION = "Face Detection";
    private static final String TEXT_DETECTION = "Text Detection";
    private static final String BARCODE_DETECTION = "Barcode Detection";
    private static final String IMAGE_LABEL_DETECTION = "Label Detection";
    private static final String CLASSIFICATION_QUANT = "Classification (quantized)";
    private static final String CLASSIFICATION_FLOAT = "Classification (float)";
    private static final String FACE_CONTOUR = "Face Contour";
    private static final String TAG = "LivePreviewActivity";
    private static final int PERMISSION_REQUESTS = 1;

    private String name;
    private CameraSource cameraSource = null;
    private CameraSourcePreview preview;
    private GraphicOverlay graphicOverlay;
    private String selectedModel = FACE_CONTOUR;
    //private PersonRecognizer mFaceRecognzier = null;

    private FaceRec mFaceRec;

    private Handler mBackgroundHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_main);

        preview = (CameraSourcePreview) findViewById(R.id.firePreview);
        if (preview == null) {
            Log.d(TAG, "Preview is null");
        }
        graphicOverlay = (GraphicOverlay) findViewById(R.id.fireFaceOverlay);
        if (graphicOverlay == null) {
            Log.d(TAG, "graphicOverlay is null");
        }

        Spinner spinner = (Spinner) findViewById(R.id.spinner);
        List<String> options = new ArrayList<>();
        options.add(FACE_CONTOUR);
        options.add(FACE_DETECTION);
        // Creating adapter for spinner
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<String>(this, R.layout.spinner_style, options);
        // Drop down layout style - list view with radio button
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // attaching data adapter to spinner
        spinner.setAdapter(dataAdapter);
        spinner.setOnItemSelectedListener(this);

        ToggleButton facingSwitch = (ToggleButton) findViewById(R.id.facingSwitch);
        facingSwitch.setChecked(true);
        facingSwitch.setOnCheckedChangeListener(this);
        // Hide the toggle button if there is only 1 camera
        if (Camera.getNumberOfCameras() == 1) {
            facingSwitch.setVisibility(View.GONE);
        }

        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
        } else {
            getRuntimePermissions();
        }

        //Initiliaze face recognizer
        new initRecAsync(null).execute();
        Button addButton = (Button) findViewById(R.id.add);
        Button recogButton = (Button) findViewById(R.id.recog);

        addButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                showAlertDialog();
            }
        });
        recogButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                MainActivity.this.onPause();
                //Get current camera frame
                Log.d(TAG, "Getting camera frmae");
                List<FirebaseVisionFace> faces = cameraSource.getMachineLearningFrameProcessor().getFaces();
                ByteBuffer latestImage = cameraSource.getMachineLearningFrameProcessor().getLatestImage();
                FrameMetadata latestFrameMetadata = cameraSource.getMachineLearningFrameProcessor().getLatestImageMetadata();

                if (!faces.isEmpty() && latestImage != null) {
                    Bitmap bitmap = BitmapUtils.getBitmap(latestImage, latestFrameMetadata);
                    Rect bbox = faces.get(0).getBoundingBox();
                    Log.d(TAG, bbox.left + " " + bbox.top + " " + bbox.width() + " " + bbox.height());
                    Bitmap crop = Bitmap.createBitmap(bitmap, bbox.left, bbox.top,
                            Math.min(bbox.width(), bitmap.getWidth()), Math.min(bbox.height(), bitmap.getHeight()));

                    new recognizeAsync(crop).execute();
                } else {
                    TextView textView = (TextView) findViewById(R.id.text);
                    textView.setText("No person found!");
                    MainActivity.this.onResume();
                }
            }
        });
    }

    private void showAlertDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter name");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);// | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        builder.setView(input);

        //Get current camera frame
        final List<FirebaseVisionFace> faces = cameraSource.getMachineLearningFrameProcessor().getFaces();
        ByteBuffer latestImage = cameraSource.getMachineLearningFrameProcessor().getLatestImage();
        FrameMetadata latestFrameMetadata = cameraSource.getMachineLearningFrameProcessor().getLatestImageMetadata();
        final Bitmap bitmap = BitmapUtils.getBitmap(latestImage, latestFrameMetadata);

        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (faces.isEmpty()) {
                    dialog.cancel();
                } else {
                    Rect bbox = faces.get(0).getBoundingBox();
                    Bitmap crop = Bitmap.createBitmap(bitmap, bbox.left, bbox.top, bbox.width(), bbox.height());

                    MainActivity.this.onPause();
                    name = input.getText().toString();
                    System.out.println("INPUT: " + name);

                    new initRecAsync(crop).execute();
                }
            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
                System.out.println("INPUT: canceld");
            }
        });

        builder.show();
    }

    private class initRecAsync extends AsyncTask<Void, Void, Void> {
        ProgressDialog dialog = new ProgressDialog(MainActivity.this);
        Bitmap bitmap;
        boolean write;

        public initRecAsync(Bitmap bitmap) {
            write = (bitmap != null);
            this.bitmap = bitmap;
        }

        @Override
        protected void onPreExecute() {
            Log.d(TAG, "initRecAsync onPreExecute called");
            dialog.setMessage("Initializing...");
            dialog.setCancelable(false);
            dialog.show();
            super.onPreExecute();
        }

        protected Void doInBackground(Void... args) {
            // create dlib_rec_example directory in sd card and copy model files
            Log.d(TAG, "Doing in background");
            mFaceRec = new FaceRec(Constants.getDLibDirectoryPath());
            File folder = new File(Constants.getDLibDirectoryPath());
            boolean success = false;
            if (!folder.exists()) {
                success = folder.mkdirs();
            }
            Log.d(TAG, "DLIB directory exists");

            if(write) {
                if (success) {
                    File image_folder = new File(Constants.getDLibImageDirectoryPath());
                    image_folder.mkdirs();
                    if (!new File(Constants.getFaceShapeModelPath()).exists()) {
                        FileUtils.copyFileFromRawToOthers(MainActivity.this, R.raw.shape_predictor_5_face_landmarks, Constants.getFaceShapeModelPath());
                    }
                    if (!new File(Constants.getFaceDescriptorModelPath()).exists()) {
                        FileUtils.copyFileFromRawToOthers(MainActivity.this, R.raw.dlib_face_recognition_resnet_model_v1, Constants.getFaceDescriptorModelPath());
                    }
                } else {
                    Log.d(TAG, "error in setting dlib_rec_example directory");
                }
                Log.d(TAG, "Creating Face Recog" + Constants.getDLibDirectoryPath());

                // Add Person
                Log.d(TAG, "Writing image");
                String targetPath = Constants.getDLibImageDirectoryPath() + "/" + name + ".jpg";
                FileOutputStream out = null;
                try {
                    out = new FileOutputStream(targetPath);
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
                    // PNG is a lossless format, the compression factor (100) is ignored
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        if (out != null) {
                            out.close();
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

            mFaceRec.train();
            return null;
        }

        protected void onPostExecute(Void result) {
            if(dialog != null && dialog.isShowing()){
                dialog.dismiss();
            }
            MainActivity.this.onResume();
        }
    }

    private class recognizeAsync extends AsyncTask<Void, Void, ArrayList<String>> {

        ProgressDialog dialog = new ProgressDialog(MainActivity.this);
        private boolean mIsComputing = false;
        Bitmap bitmap;

        public recognizeAsync(Bitmap bitmap) {
            this.bitmap = bitmap;
        }

        @Override
        protected void onPreExecute() {
            dialog.setMessage("Recognizing...");
            dialog.setCancelable(false);
            dialog.show();
            super.onPreExecute();
        }

        @Override
        protected ArrayList<String> doInBackground(Void... voids) {
            ArrayList<String> names = new ArrayList<>();

            try {
                long startTime = System.currentTimeMillis();
                //mFaceRec = new FaceRec(Constants.getDLibDirectoryPath());
                List<VisionDetRet> results = MainActivity.this.mFaceRec.recognize(bitmap);

                for (VisionDetRet n : results) {
                    names.add(n.getLabel());
                }

                long endTime = System.currentTimeMillis();
                Log.d(TAG, "Time cost: " + String.valueOf((endTime - startTime) / 1000f) + " sec");
            } catch (NullPointerException ex){
                Log.d(TAG, "No bitmap!");
                throw ex;
            }

            return names;
        }

        protected void onPostExecute(ArrayList<String> names) {
            if(dialog != null && dialog.isShowing()){
                dialog.dismiss();
                AlertDialog.Builder builder1 = new AlertDialog.Builder(MainActivity.this);
                String result = getResultMessage(names);
                builder1.setMessage(result);
                builder1.setCancelable(true);
                AlertDialog alert11 = builder1.create();
                alert11.show();

                //Set Text
                TextView textView = (TextView) findViewById(R.id.text);
                textView.setText(result);
                MainActivity.this.onResume();
            }
        }
    }

    private String getResultMessage(ArrayList<String> names) {
        String msg = new String();
        if (names.isEmpty()) {
            msg = "No face detected or Unknown person";

        } else {
            for(int i=0; i<names.size(); i++) {
                msg += names.get(i).split(Pattern.quote("."))[0];
                if (i!=names.size()-1) msg+=", ";
            }
            msg+=" found!";
        }
        return msg;
    }

    @Override
    public synchronized void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        selectedModel = parent.getItemAtPosition(pos).toString();
        Log.d(TAG, "Selected model: " + selectedModel);
        preview.stop();
        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
            cameraSource.setFacing(CameraSource.CAMERA_FACING_FRONT);
            startCameraSource();
        } else {
            getRuntimePermissions();
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Do nothing.
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        Log.d(TAG, "Set facing");
        if (cameraSource != null) {
            if (isChecked) {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_FRONT);
            } else {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_BACK);
            }
        }
        preview.stop();
        startCameraSource();
    }

    private void createCameraSource(String model) {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = new CameraSource(this, graphicOverlay);
        }

        switch (model) {
            case FACE_DETECTION:
                Log.i(TAG, "Using Face Detector Processor");
                cameraSource.setMachineLearningFrameProcessor(new FaceDetectionProcessor());
                break;
            case FACE_CONTOUR:
                Log.i(TAG, "Using Face Contour Detector Processor");
                cameraSource.setMachineLearningFrameProcessor(new FaceContourDetectorProcessor());
                //cameraSource.setMachineLearningFrameProcessor(new FaceContourDetectorProcessor(mFaceRecognzier));
                break;
            default:
                break;
        }
    }

    /**
     * Starts or restarts the camera source, if it exists. If the camera source doesn't exist yet
     * (e.g., because onResume was called before the camera source was created), this will be called
     * again when the camera source is created.
     */
    private void startCameraSource() {
        if (cameraSource != null) {
            try {
                if (preview == null) {
                    Log.d(TAG, "resume: Preview is null");
                }
                if (graphicOverlay == null) {
                    Log.d(TAG, "resume: graphOverlay is null");
                }
                preview.start(cameraSource, graphicOverlay);

            } catch (IOException e) {
                Log.e(TAG, "Unable to start camera source.", e);
                cameraSource.release();
                cameraSource = null;
            }
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        startCameraSource();
    }

    /** Stops the camera. */
    @Override
    protected void onPause() {
        super.onPause();
        preview.stop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraSource != null) {
            cameraSource.release();
        }
        if (mBackgroundHandler != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
                mBackgroundHandler.getLooper().quitSafely();
            } else {
                mBackgroundHandler.getLooper().quit();
            }
            mBackgroundHandler = null;
        }
    }

    private Handler getBackgroundHandler() {
        if (mBackgroundHandler == null) {
            HandlerThread thread = new HandlerThread("background");
            thread.start();
            mBackgroundHandler = new Handler(thread.getLooper());
        }
        return mBackgroundHandler;
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    this.getPackageManager()
                            .getPackageInfo(this.getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                return false;
            }
        }
        return true;
    }

    private void getRuntimePermissions() {
        List<String> allNeededPermissions = new ArrayList<>();
        for (String permission : getRequiredPermissions()) {
            if (!isPermissionGranted(this, permission)) {
                allNeededPermissions.add(permission);
            }
        }

        if (!allNeededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(
                    this, allNeededPermissions.toArray(new String[0]), PERMISSION_REQUESTS);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        Log.i(TAG, "Permission granted!");
        if (allPermissionsGranted()) {
            createCameraSource(selectedModel);
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private static boolean isPermissionGranted(Context context, String permission) {
        if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Permission granted: " + permission);
            return true;
        }
        Log.i(TAG, "Permission NOT granted: " + permission);
        return false;
    }
}
