"""
Hand Tracking & Spatial Continuity Module.
Locks onto the active hand target, prevents abrupt jumps between hands,
and provides graceful tracking-loss recovery.
"""

import math
import time
from enum import Enum
from typing import List, Optional, Tuple

from src.detection.hand_detector import DetectedHand, HandLandmark, INDEX_TIP, WRIST
from src.detection.one_euro_filter import OneEuroFilter2D
from src.utils.logging_config import setup_logger

logger = setup_logger("HandTracker")


class TrackingStatus(Enum):
    TRACKING = "Tracking"
    SEARCHING = "Searching"
    LOST = "Lost"


class HandTracker:
    """
    Maintains spatial continuity for the primary controlled hand.
    Eliminates target-swapping glitches and jitter.
    """

    def __init__(
        self,
        preferred_handedness: str = "Right",
        control_point: str = "index_tip",
        tracking_loss_timeout: float = 0.14,
        min_cutoff: float = 1.2,
        beta: float = 15.0,
        sensitivity: float = 1.0,
        is_mirrored: bool = True
    ):
        """
        Args:
            preferred_handedness: "Right", "Left", or "Any".
            control_point: "index_tip" (default) or "palm_center".
            tracking_loss_timeout: Seconds to hold position when hand is momentarily lost.
            min_cutoff: One Euro filter min cutoff frequency (Hz).
            beta: One Euro filter speed coefficient.
            sensitivity: Cursor sensitivity multiplier.
            is_mirrored: Whether camera feed is mirrored.
        """
        self.preferred_handedness = preferred_handedness
        self.control_point = control_point
        self.tracking_loss_timeout = tracking_loss_timeout
        self.sensitivity = max(0.5, min(3.0, sensitivity))
        self.is_mirrored = is_mirrored

        self.filter = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta)
        self.status = TrackingStatus.LOST

        self.last_hand: Optional[DetectedHand] = None
        self.last_raw_pos: Optional[Tuple[float, float]] = None
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_filtered_pos: Optional[Tuple[float, float]] = None
        self._prev_landmarks: Optional[List[HandLandmark]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_seen_time: float = 0.0
        self.current_confidence: float = 0.0

    def update_settings(
        self,
        control_point: Optional[str] = None,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        timeout: Optional[float] = None,
        sensitivity: Optional[float] = None,
        is_mirrored: Optional[bool] = None
    ) -> None:
        """Updates runtime tracker configuration."""
        if control_point:
            self.control_point = control_point
        if min_cutoff is not None and beta is not None:
            self.filter.set_parameters(min_cutoff, beta)
        if timeout is not None:
            self.tracking_loss_timeout = timeout
        if sensitivity is not None:
            self.sensitivity = max(0.5, min(3.0, sensitivity))
        if is_mirrored is not None:
            self.is_mirrored = is_mirrored

    def _extract_control_point(self, hand: DetectedHand) -> Tuple[float, float]:
        """Extracts normalized (x, y) coordinates for the selected control target."""
        if self.control_point == "palm_center":
            return hand.center
        # Default: Index Fingertip
        return hand.index_tip

    def update(self, detected_hands: List[DetectedHand], now: Optional[float] = None) -> Tuple[TrackingStatus, Optional[DetectedHand], Optional[Tuple[float, float]]]:
        """
        Processes detected hands and returns stabilized target and cursor coordinates.
        
        Returns:
            Tuple[TrackingStatus, Optional[DetectedHand], Optional[Tuple[float, float]]]:
            (Tracking status, Locked DetectedHand or None, Filtered (x, y) or None)
        """
        current_time = now if now is not None else time.time()

        if detected_hands:
            # 1. Select the best matching hand using raw camera frame space
            best_hand = self._select_primary_hand(detected_hands)

            raw_ctrl_x, raw_ctrl_y = self._extract_control_point(best_hand)

            # Compute raw velocity for predictive continuity
            if self.last_raw_pos is not None and self.last_seen_time > 0:
                dt_step = max(1e-4, current_time - self.last_seen_time)
                vx = (raw_ctrl_x - self.last_raw_pos[0]) / dt_step
                vy = (raw_ctrl_y - self.last_raw_pos[1]) / dt_step
                self._velocity = (vx, vy)
            else:
                self._velocity = (0.0, 0.0)

            # Adaptive smoothing across all 21 landmarks: zero lag during fast motion, stable when still
            speed = math.hypot(self._velocity[0], self._velocity[1])
            alpha = max(0.40, min(0.92, 0.50 + speed * 0.40))
            if self._prev_landmarks and len(self._prev_landmarks) == len(best_hand.landmarks):
                smoothed_lms = []
                for curr_lm, prev_lm in zip(best_hand.landmarks, self._prev_landmarks):
                    sx = alpha * curr_lm.x + (1.0 - alpha) * prev_lm.x
                    sy = alpha * curr_lm.y + (1.0 - alpha) * prev_lm.y
                    sz = alpha * curr_lm.z + (1.0 - alpha) * prev_lm.z
                    smoothed_lms.append(HandLandmark(x=sx, y=sy, z=sz))
                best_hand.landmarks = smoothed_lms
            self._prev_landmarks = best_hand.landmarks

            # Prepare cursor coordinate mapping
            raw_x, raw_y = self._extract_control_point(best_hand)

            # If camera preview is NOT mirrored, invert horizontal coordinate so physical right moves cursor right
            if not self.is_mirrored:
                raw_x = 1.0 - raw_x

            # Apply configurable cursor sensitivity centered at (0.5, 0.5)
            if self.sensitivity != 1.0:
                raw_x = 0.5 + (raw_x - 0.5) * self.sensitivity
                raw_y = 0.5 + (raw_y - 0.5) * self.sensitivity

            raw_x = max(0.0, min(1.0, raw_x))
            raw_y = max(0.0, min(1.0, raw_y))

            # Apply adaptive One Euro filter
            filtered_x, filtered_y = self.filter.filter(raw_x, raw_y, current_time)
            # Clamp in normalized range [0.0, 1.0]
            filtered_x = max(0.0, min(1.0, filtered_x))
            filtered_y = max(0.0, min(1.0, filtered_y))

            self.last_hand = best_hand
            self.last_raw_pos = (raw_ctrl_x, raw_ctrl_y)
            self.last_pos = (raw_x, raw_y)
            self.last_filtered_pos = (filtered_x, filtered_y)
            self.last_seen_time = current_time
            self.status = TrackingStatus.TRACKING
            self.current_confidence = best_hand.confidence

            return self.status, best_hand, self.last_filtered_pos

        else:
            # 2. Handle absence of hand detection with velocity extrapolation
            elapsed = current_time - self.last_seen_time
            if self.last_filtered_pos is not None and elapsed < self.tracking_loss_timeout:
                self.status = TrackingStatus.SEARCHING
                dt_step = min(0.04, max(1e-4, current_time - self.last_seen_time if self.last_seen_time > 0 else 0.033))

                # Extrapolate cursor position using tracked velocity
                vx, vy = self._velocity
                vx *= 0.85  # Gentle inertia decay
                vy *= 0.85
                self._velocity = (vx, vy)

                dx_screen = (-vx if not self.is_mirrored else vx) * self.sensitivity * dt_step
                dy_screen = vy * self.sensitivity * dt_step

                extrap_x = max(0.0, min(1.0, self.last_filtered_pos[0] + dx_screen))
                extrap_y = max(0.0, min(1.0, self.last_filtered_pos[1] + dy_screen))
                self.last_filtered_pos = (extrap_x, extrap_y)

                # Extrapolate all 21 landmarks so skeleton remains connected and continuous
                if self.last_hand and self.last_hand.landmarks:
                    dx_raw = vx * dt_step
                    dy_raw = vy * dt_step
                    extrap_lms = []
                    for lm in self.last_hand.landmarks:
                        ex = max(0.0, min(1.0, lm.x + dx_raw))
                        ey = max(0.0, min(1.0, lm.y + dy_raw))
                        extrap_lms.append(HandLandmark(x=ex, y=ey, z=lm.z))
                    self.last_hand.landmarks = extrap_lms
                    bx, by, bw, bh = self.last_hand.bbox
                    self.last_hand.bbox = (int(bx + dx_raw * 640), int(by + dy_raw * 480), bw, bh)

                return self.status, self.last_hand, self.last_filtered_pos

            # Hand lost beyond grace timeout
            self.status = TrackingStatus.LOST
            self.current_confidence = 0.0
            self.filter.reset()
            self.last_hand = None
            self.last_pos = None
            self.last_raw_pos = None
            self._prev_landmarks = None
            self._velocity = (0.0, 0.0)
            self.last_filtered_pos = None
            return self.status, None, None

    def _select_primary_hand(self, detected_hands: List[DetectedHand]) -> DetectedHand:
        """
        Picks the most consistent hand:
        1. If previously tracking, chooses hand closest to previous raw position.
        2. Otherwise, prefers preferred handedness with highest confidence.
        """
        if len(detected_hands) == 1:
            return detected_hands[0]

        anchor_pos = self.last_raw_pos if self.last_raw_pos is not None else self.last_pos
        if anchor_pos is not None:
            # Spatial distance heuristic in matching coordinate space
            best_hand = detected_hands[0]
            min_dist = float("inf")
            for hand in detected_hands:
                hx, hy = self._extract_control_point(hand)
                dist = math.hypot(hx - anchor_pos[0], hy - anchor_pos[1])
                # Small bonus for matching handedness
                if self.preferred_handedness != "Any" and hand.handedness == self.preferred_handedness:
                    dist *= 0.85
                if dist < min_dist:
                    min_dist = dist
                    best_hand = hand
            return best_hand

        # First acquisition: match preferred handedness if possible
        if self.preferred_handedness != "Any":
            for hand in detected_hands:
                if hand.handedness == self.preferred_handedness:
                    return hand

        # Fallback to highest confidence
        return max(detected_hands, key=lambda h: h.confidence)

    def reset(self) -> None:
        """Resets tracker state."""
        self.status = TrackingStatus.LOST
        self.filter.reset()
        self.last_hand = None
        self.last_pos = None
        self.last_raw_pos = None
        self._prev_landmarks = None
        self._velocity = (0.0, 0.0)
        self.last_filtered_pos = None
        self.last_seen_time = 0.0
        self.current_confidence = 0.0
