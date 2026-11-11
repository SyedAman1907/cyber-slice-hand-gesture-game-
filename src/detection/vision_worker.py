"""
Asynchronous Vision Pipeline Worker.
Decouples camera I/O and neural network inference from the 60 FPS Pygame render loop.
Ensures zero-latency frame delivery, real-time FPS telemetry, and thread-safe state sharing.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.detection.hand_detector import BaseHandDetector, DetectedHand
from src.detection.hand_tracker import HandTracker, TrackingStatus
from src.gestures.gesture_recognizer import GestureRecognizer, RecognizedGesture
from src.utils.logging_config import setup_logger

logger = setup_logger("VisionWorker")


@dataclass
class VisionState:
    """Thread-safe snapshot of the latest computer vision inference."""
    raw_frame: Optional[np.ndarray] = None
    detected_hands: List[DetectedHand] = field(default_factory=list)
    primary_hand: Optional[DetectedHand] = None
    cursor_pos: Tuple[float, float] = (0.5, 0.5)
    gesture: RecognizedGesture = field(default_factory=lambda: RecognizedGesture(
        name="Unknown", confidence=0.0, description="Scanning...",
        action="none", cursor_pos=(0.5, 0.5)
    ))
    tracking_status: TrackingStatus = TrackingStatus.LOST
    camera_fps: float = 30.0
    processing_fps: float = 30.0
    inference_time_ms: float = 0.0
    gesture_time_ms: float = 0.0
    capture_timestamp: float = 0.0
    hand_confidence: float = 0.0


class VisionWorker:
    """
    Background worker thread continuously executing detection,
    hand tracking, One Euro filtering, and gesture recognition.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        detector: BaseHandDetector,
        hand_tracker: HandTracker,
        gesture_recognizer: GestureRecognizer
    ):
        self.camera = camera_manager
        self.detector = detector
        self.tracker = hand_tracker
        self.gesture_recognizer = gesture_recognizer

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest shared state
        self._latest_state = VisionState()

        # Telemetry
        self._proc_fps_history: List[float] = []
        self._last_proc_time = time.time()

    def start(self) -> None:
        """Starts background vision processing worker."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, name="VisionWorkerThread", daemon=True)
        self._thread.start()
        logger.info("Vision processing worker started on background thread.")

    def stop(self) -> None:
        """Stops background worker thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        logger.info("Vision processing worker stopped.")

    def _process_loop(self) -> None:
        """Internal worker loop running at camera frame arrival rate with zero duplicate inference."""
        prev_time = time.time()
        last_frame_id = -1

        while self._running:
            success, frame, cap_timestamp, frame_id = self.camera.get_latest_frame_blocking(
                last_frame_id=last_frame_id, timeout=0.033
            )
            if not success or frame is None:
                continue
            last_frame_id = frame_id

            t_now = time.time()
            dt = t_now - prev_time
            prev_time = t_now
            if dt > 0:
                inst_fps = 1.0 / dt
                self._proc_fps_history.append(inst_fps)
                if len(self._proc_fps_history) > 30:
                    self._proc_fps_history.pop(0)
            proc_fps = sum(self._proc_fps_history) / len(self._proc_fps_history) if self._proc_fps_history else 30.0

            # 1. Run Hand Detection
            t_inf_start = time.perf_counter()
            detected_hands = self.detector.detect(frame)
            inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

            # 2. Run Spatial Target Tracker & Adaptive One Euro Filter
            tracking_status, primary_hand, filtered_cursor = self.tracker.update(
                detected_hands, now=cap_timestamp
            )

            # 3. Classify Gesture on Tracked Primary Hand
            t_gest_start = time.perf_counter()
            if primary_hand is not None:
                gesture = self.gesture_recognizer.recognize(primary_hand)
                # Ensure cursor position uses One Euro filtered coordinates
                if filtered_cursor is not None:
                    gesture.cursor_pos = filtered_cursor
                hand_conf = primary_hand.confidence
            else:
                if tracking_status == TrackingStatus.LOST:
                    self.gesture_recognizer.reset()
                gesture = RecognizedGesture(
                    name="Unknown", confidence=0.0, description="Hand not detected",
                    action="none", cursor_pos=(0.5, 0.5)
                )
                hand_conf = 0.0
            gest_time_ms = (time.perf_counter() - t_gest_start) * 1000.0

            cursor_out = filtered_cursor if filtered_cursor is not None else (0.5, 0.5)

            # Ensure continuous hand rendering even during brief detector dropouts
            display_hands = detected_hands if detected_hands else ([primary_hand] if primary_hand is not None else [])

            # 4. Atomic state publish
            with self._lock:
                self._latest_state = VisionState(
                    raw_frame=frame,
                    detected_hands=display_hands,
                    primary_hand=primary_hand,
                    cursor_pos=cursor_out,
                    gesture=gesture,
                    tracking_status=tracking_status,
                    camera_fps=self.camera.current_fps,
                    processing_fps=proc_fps,
                    inference_time_ms=inf_time_ms,
                    gesture_time_ms=gest_time_ms,
                    capture_timestamp=cap_timestamp,
                    hand_confidence=hand_conf
                )

    def get_latest_state(self) -> VisionState:
        """Returns the freshest vision state snapshot without blocking."""
        with self._lock:
            return self._latest_state
