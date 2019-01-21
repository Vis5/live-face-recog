// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.themon.test.mlkit_faciallandmarks.common;

import android.graphics.Bitmap;

import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;

import java.nio.ByteBuffer;
import java.util.List;

/** An inferface to process the images with different ML Kit detectors and custom image models. */
public interface VisionImageProcessor {

  ByteBuffer getLatestImage();
  FrameMetadata getLatestImageMetadata();
  List<FirebaseVisionFace> getFaces();

  /** Processes the images with the underlying machine learning models. */
  void process(ByteBuffer data, FrameMetadata frameMetadata, GraphicOverlay graphicOverlay)
      throws FirebaseMLException;

  /** Processes the bitmap images. */
  void process(Bitmap bitmap, GraphicOverlay graphicOverlay);

  /** Stops the underlying machine learning model and release resources. */
  void stop();
}
