# Hand Detection Model Directory

This directory stores deep learning model weights for the YOLO hand detection backend.

## Detection Backends Supported

The project features a modular multi-backend detection architecture:
1. **MediaPipe Hands (Primary & Default)**:
   - Tracks 21 3D hand landmarks per hand with high accuracy and low CPU latency.
   - Built into Google MediaPipe (Apache 2.0 license).
   - Requires zero external manual weight downloads.
   - Provides full skeletal tracking, left/right hand classification, and normalized 3D coordinates.

2. **YOLO Hand Detection (Optional / Configurable)**:
   - Ultralytics YOLOv8 architecture (AGPL-3.0).
   - Standard COCO pretrained weights (`yolov8n.pt`) contain 80 everyday classes (person, car, dog, etc.), but **do NOT include a dedicated hand class**.
   - To use YOLO hand detection, a model trained on a hand dataset (e.g. Egohands, Oxford Hand Dataset, or Roboflow Universe Hand Keypoint/Detection) is required.
   - Place your trained YOLO hand model file here as:
     `models/yolov8n-hand.pt`

## Configuring the Backend

In `config/settings.json`, set:
```json
{
  "detection": {
    "backend": "mediapipe", // or "yolo"
    "yolo_model_path": "models/yolov8n-hand.pt",
    "yolo_confidence_threshold": 0.5
  }
}
```

If `yolo` is selected but the weight file is absent or fails to load, the system automatically falls back to MediaPipe Hands with a clear diagnostic log.
