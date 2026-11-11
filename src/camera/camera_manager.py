"""
Module 1: Camera Input Module.
Provides robust, threaded, non-blocking webcam capture with device selection,
FPS calculation, mirror toggling, and clean resource cleanup.
"""

import time
import threading
from enum import Enum
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("CameraManager")


class CameraStatus(Enum):
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    ERROR = "Error"


class CameraManager:
    """
    Thread-safe webcam capture manager using OpenCV.
    Runs frame capture in a background daemon thread to avoid blocking the game or UI loop.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirrored: bool = True,
        auto_reconnect: bool = True
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mirrored = mirrored
        self.auto_reconnect = auto_reconnect

        self.cap: Optional[cv2.VideoCapture] = None
        self.status = CameraStatus.DISCONNECTED
        self.error_message: str = ""

        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._new_frame_cond = threading.Condition(self._lock)

        # Frame buffer
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._frame_count = 0

        # FPS tracking
        self.current_fps: float = 0.0
        self._fps_history: List[float] = []
        self._last_fps_calc_time = time.time()

    def start(self) -> bool:
        """Starts the camera and begins the background capture thread."""
        with self._lock:
            if self._running:
                logger.info("Camera is already running.")
                return True

            logger.info("Opening camera device %d (%dx%d @ %d FPS)...",
                        self.device_index, self.width, self.height, self.target_fps)
            self.status = CameraStatus.CONNECTING
            self.error_message = ""

        # Attempt to open device
        if not self._open_device():
            with self._lock:
                self.status = CameraStatus.ERROR
                self.error_message = f"Unable to open camera device index {self.device_index}"
            logger.error(self.error_message)
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()

        with self._lock:
            self.status = CameraStatus.CONNECTED
        logger.info("Camera successfully started and streaming.")
        return True

    def _open_device(self) -> bool:
        """Initializes the cv2.VideoCapture object with DirectShow on Windows for fast startup."""
        try:
            # On Windows, cv2.CAP_DSHOW improves startup time and reliability
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.device_index, backend)
            if not self.cap or not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                return False

            # Request MJPG for high-framerate streaming if supported
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # Minimize driver queue buffering for zero-lag real-time capture
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error("Exception opening camera device %d: %s", self.device_index, e)
            return False

    def _capture_loop(self) -> None:
        """Internal capture loop running in background thread without artificial sleep."""
        prev_time = time.time()
        fail_count = 0
        while self._running:
            if not self.cap or not self.cap.isOpened():
                if self.auto_reconnect:
                    logger.warning("Camera disconnected. Retrying in 1 second...")
                    time.sleep(1.0)
                    if self._open_device():
                        logger.info("Reconnected to camera device %d.", self.device_index)
                        continue
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            now = time.time()
            if not ret or frame is None:
                fail_count += 1
                if fail_count == 40:
                    logger.warning("Camera frames are failing to read. Check if another app is using the camera or check camera privacy permissions.")
                time.sleep(0.01)
                continue
            fail_count = 0

            if self.mirrored:
                frame = cv2.flip(frame, 1)

            # Update FPS calculation
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_history.append(inst_fps)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                self.current_fps = sum(self._fps_history) / len(self._fps_history)

            with self._lock:
                self._latest_frame = frame
                self._frame_timestamp = now
                self._frame_count += 1
                self._new_frame_cond.notify_all()

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the freshest captured frame.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (Success flag, frame as BGR numpy array)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get_frame_with_timestamp(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Retrieves freshest frame alongside its capture timestamp for latency calculation."""
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._frame_timestamp

    def get_latest_frame_blocking(
        self,
        last_frame_id: int,
        timeout: float = 0.05
    ) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """
        Blocks until a strictly newer frame arrives or timeout expires.
        Prevents redundant inference and eliminates CPU spin.
        
        Returns:
            Tuple[bool, Optional[np.ndarray], float, int]: (Success, Frame copy, Timestamp, Frame Count)
        """
        with self._new_frame_cond:
            if self._running and self._frame_count <= last_frame_id:
                self._new_frame_cond.wait(timeout=timeout)
            if self._latest_frame is None or self._frame_count <= last_frame_id:
                return False, None, self._frame_timestamp, self._frame_count
            return True, self._latest_frame.copy(), self._frame_timestamp, self._frame_count

    def stop(self) -> None:
        """Stops capture loop and releases the camera resource."""
        self._running = False
        with self._new_frame_cond:
            self._new_frame_cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

        with self._lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.status = CameraStatus.DISCONNECTED
            self._latest_frame = None

        logger.info("Camera device %d stopped.", self.device_index)

    def retry(self) -> bool:
        """Stops and attempts to restart camera capture."""
        logger.info("Retrying camera initialization...")
        self.stop()
        time.sleep(0.5)
        return self.start()

    def set_mirrored(self, mirrored: bool) -> None:
        """Toggles horizontal mirror preview."""
        self.mirrored = mirrored

    def set_resolution(self, width: int, height: int) -> None:
        """Updates capture resolution and applies to current stream if open."""
        self.width = width
        self.height = height
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_device_index(self, index: int) -> bool:
        """Switches to another camera device index."""
        if self.device_index == index and self.status == CameraStatus.CONNECTED:
            return True
        self.device_index = index
        if self._running:
            return self.retry()
        return True

    @staticmethod
    def list_available_cameras(max_tested: int = 4) -> List[int]:
        """
        Scans for available camera indices.
        
        Args:
            max_tested: Maximum number of device indices to probe.
            
        Returns:
            List[int]: List of available camera indices.
        """
        available: List[int] = []
        for idx in range(max_tested):
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            temp_cap = cv2.VideoCapture(idx, backend)
            if temp_cap and temp_cap.isOpened():
                available.append(idx)
                temp_cap.release()
        return available if available else [0]
