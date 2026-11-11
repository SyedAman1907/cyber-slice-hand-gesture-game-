"""
MediaPipe Hands Detection Implementation.
Extracts 21 3D hand landmarks per detected hand with sub-millisecond tracking.
"""

import time
from typing import List, Optional
import cv2
import numpy as np

from src.detection.hand_detector import (
    BaseHandDetector,
    DetectedHand,
    HandLandmark,
    LandmarkSmoother,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("MediaPipeDetector")


class MediaPipeHandDetector(BaseHandDetector):
    """Hand detector utilizing Google MediaPipe Hands pipeline."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.45,
        model_complexity: int = 0,
        is_mirrored: bool = True,
        smooth_landmarks: bool = False
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.is_mirrored = is_mirrored
        self.smooth_landmarks = smooth_landmarks

        self.last_inference_time_ms: float = 0.0
        self._mp_hands = None
        self._hands = None
        self._initialized = False

        self._init_mediapipe()

    def set_mirrored(self, mirrored: bool) -> None:
        """Updates mirror mode for handedness inversion."""
        self.is_mirrored = mirrored

    def update_confidences(
        self,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None
    ) -> None:
        """Updates detection/tracking confidences and re-initializes graph if needed."""
        changed = False
        if min_detection_confidence is not None and abs(min_detection_confidence - self.min_detection_confidence) > 1e-3:
            self.min_detection_confidence = min_detection_confidence
            changed = True
        if min_tracking_confidence is not None and abs(min_tracking_confidence - self.min_tracking_confidence) > 1e-3:
            self.min_tracking_confidence = min_tracking_confidence
            changed = True
        if changed:
            self.release()
            self._init_mediapipe()

    def _init_mediapipe(self) -> bool:
        """Initializes the MediaPipe Hands graph."""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity
            )
            self._initialized = True
            logger.info("MediaPipe Hands initialized (complexity=%d, max_hands=%d, det_conf=%.2f, track_conf=%.2f).",
                        self.model_complexity, self.max_num_hands, self.min_detection_confidence, self.min_tracking_confidence)
            return True
        except Exception as e:
            logger.error("Failed to initialize MediaPipe Hands: %s", e)
            self._initialized = False
            return False

    def detect(self, frame: np.ndarray) -> List[DetectedHand]:
        """
        Processes a BGR video frame and returns detected hands with normalized landmarks.
        
        Args:
            frame: Input image frame as BGR numpy array.
            
        Returns:
            List[DetectedHand]: Detected hands with landmark points and bounding boxes.
        """
        if not self._initialized or self._hands is None:
            if not self._init_mediapipe():
                return []

        if frame is None or frame.size == 0:
            return []

        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        # Optimize inference: if camera resolution is high, downscale for MediaPipe inference
        if w > 640:
            scale = 640.0 / w
            inf_h = int(h * scale)
            inf_frame = cv2.resize(frame, (640, inf_h), interpolation=cv2.INTER_LINEAR)
        else:
            inf_frame = frame

        # MediaPipe expects RGB images
        rgb = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # Performance optimization

        try:
            results = self._hands.process(rgb)
        except Exception as e:
            logger.error("Error during MediaPipe inference: %s", e)
            return []
        finally:
            self.last_inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        detected_hands: List[DetectedHand] = []

        if not results.multi_hand_landmarks:
            return detected_hands

        # Extract handedness if available
        handedness_list = []
        if results.multi_handedness:
            for hand_classification in results.multi_handedness:
                raw_label = hand_classification.classification[0].label
                score = hand_classification.classification[0].score
                # Invert label when mirrored so Left/Right corresponds to actual physical user hand
                if self.is_mirrored:
                    corrected_label = "Right" if raw_label == "Left" else "Left"
                else:
                    corrected_label = raw_label
                handedness_list.append((corrected_label, score))

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Parse 21 landmarks
            raw_landmarks: List[HandLandmark] = []
            x_coords: List[float] = []
            y_coords: List[float] = []

            for lm in hand_lms.landmark:
                # Clamp coordinates safely in [0.0, 1.0]
                clamped_x = float(np.clip(lm.x, 0.0, 1.0))
                clamped_y = float(np.clip(lm.y, 0.0, 1.0))
                raw_landmarks.append(HandLandmark(x=clamped_x, y=clamped_y, z=float(lm.z)))
                x_coords.append(clamped_x)
                y_coords.append(clamped_y)

            landmarks = raw_landmarks

            # Compute pixel bounding box
            min_x = int(min(x_coords) * w)
            max_x = int(max(x_coords) * w)
            min_y = int(min(y_coords) * h)
            max_y = int(max(y_coords) * h)

            # Add padding
            pad_x = int((max_x - min_x) * 0.15)
            pad_y = int((max_y - min_y) * 0.15)
            bx = max(0, min_x - pad_x)
            by = max(0, min_y - pad_y)
            bw = min(w - bx, (max_x - min_x) + 2 * pad_x)
            bh = min(h - by, (max_y - min_y) + 2 * pad_y)

            # Handedness label and confidence
            label = "Right"
            conf = 0.90
            if idx < len(handedness_list):
                label, conf = handedness_list[idx]

            detected_hands.append(
                DetectedHand(
                    landmarks=landmarks,
                    bbox=(bx, by, bw, bh),
                    handedness=label,
                    confidence=conf,
                    hand_id=idx
                )
            )

        return detected_hands

    def release(self) -> None:
        """Closes the MediaPipe Hands object and frees resources."""
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None
        self._initialized = False
        logger.info("MediaPipe Hands detector released.")
