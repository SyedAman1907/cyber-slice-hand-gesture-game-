"""
YOLO Hand Detection Implementation.
Integrates Ultralytics YOLO with automatic validation of class names,
detection of dedicated hand classes, and safe fallback to MediaPipe Hands.
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from src.detection.hand_detector import (
    BaseHandDetector,
    DetectedHand,
    HandLandmark,
    INDEX_TIP,
    WRIST,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("YOLODetector")


class YOLOHandDetector(BaseHandDetector):
    """
    Hand detector utilizing Ultralytics YOLOv8.
    Inspects model class names and safely falls back to MediaPipe Hands
    if the specified model is missing or lacks a dedicated hand class.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n-hand.pt",
        confidence_threshold: float = 0.5,
        fallback_to_mediapipe: bool = True
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.fallback_to_mediapipe = fallback_to_mediapipe

        self._model = None
        self._hand_class_ids: List[int] = []
        self._is_yolo_active = False
        self._fallback_detector: Optional[BaseHandDetector] = None

        self._init_detector()

    def _init_detector(self) -> None:
        """Loads and verifies YOLO model, or activates fallback."""
        if not self.model_path.exists():
            logger.warning(
                "YOLO model weights not found at '%s'. "
                "COCO models lack hand classes. Activating MediaPipe fallback.",
                self.model_path
            )
            self._activate_fallback()
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            names = self._model.names or {}

            # Verify that the model contains hand-related classes
            self._hand_class_ids = [
                idx for idx, name in names.items() if "hand" in str(name).lower()
            ]

            if not self._hand_class_ids:
                logger.warning(
                    "Loaded YOLO model '%s' contains classes %s but NO dedicated 'hand' class! "
                    "Pretrained COCO models cannot track hand gestures. Activating MediaPipe fallback.",
                    self.model_path, list(names.values())[:10]
                )
                self._activate_fallback()
                return

            self._is_yolo_active = True
            logger.info("YOLO Hand Model loaded successfully from '%s' with hand classes %s.",
                        self.model_path, self._hand_class_ids)

        except Exception as e:
            logger.error("Failed to load YOLO model from '%s': %s", self.model_path, e)
            self._activate_fallback()

    def _activate_fallback(self) -> None:
        """Instantiates MediaPipe as the reliable fallback."""
        if not self.fallback_to_mediapipe:
            self._is_yolo_active = False
            return

        logger.info("Instantiating MediaPipe fallback hand detector...")
        from src.detection.mediapipe_detector import MediaPipeHandDetector
        self._fallback_detector = MediaPipeHandDetector()
        self._is_yolo_active = False

    def detect(self, frame: np.ndarray) -> List[DetectedHand]:
        """
        Detects hands using YOLO if active, or routes to MediaPipe fallback.
        """
        if not self._is_yolo_active:
            if self._fallback_detector:
                return self._fallback_detector.detect(frame)
            return []

        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        try:
            results = self._model(frame, conf=self.confidence_threshold, verbose=False)
        except Exception as e:
            logger.error("YOLO inference failed: %s", e)
            return []

        detected_hands: List[DetectedHand] = []
        if not results or len(results) == 0:
            return detected_hands

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detected_hands

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            if cls_id not in self._hand_class_ids:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            # Normalized center and fingertips
            cx = (x1 + x2) / (2.0 * w)
            cy = (y1 + y2) / (2.0 * h)

            # Generate synthetic landmark approximation for gesture pipeline
            # with 21 points so downstream consumers don't crash
            landmarks: List[HandLandmark] = []
            for lm_idx in range(21):
                if lm_idx == INDEX_TIP:
                    # Index tip estimated at top-center of bounding box
                    landmarks.append(HandLandmark(x=cx, y=max(0.0, y1 / h)))
                elif lm_idx == WRIST:
                    landmarks.append(HandLandmark(x=cx, y=min(1.0, y2 / h)))
                else:
                    landmarks.append(HandLandmark(x=cx, y=cy))

            detected_hands.append(
                DetectedHand(
                    landmarks=landmarks,
                    bbox=(x1, y1, bw, bh),
                    handedness="Hand",
                    confidence=conf,
                    hand_id=idx
                )
            )

        return detected_hands

    def set_mirrored(self, mirrored: bool) -> None:
        """Propagates mirrored flag to fallback detector."""
        if self._fallback_detector:
            self._fallback_detector.set_mirrored(mirrored)

    def release(self) -> None:
        """Releases YOLO or fallback model resources."""
        self._model = None
        if self._fallback_detector:
            self._fallback_detector.release()
            self._fallback_detector = None
        logger.info("YOLO Hand detector released.")
