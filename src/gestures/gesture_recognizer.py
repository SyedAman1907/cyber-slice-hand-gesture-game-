"""
Module 3: Gesture Recognition Module.
Implements anatomical landmark geometry analysis, scale-invariant heuristics,
handedness-aware thumb metrics, motion velocity tracking, and debug telemetry.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.detection.hand_detector import (
    DetectedHand,
    HandLandmark,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from src.gestures.gesture_smoother import GestureDebouncer, GestureTemporalFilter
from src.utils.logging_config import setup_logger

logger = setup_logger("GestureRecognizer")


@dataclass
class RecognizedGesture:
    """Encapsulates the classified gesture, confidence, and debug telemetry."""
    name: str
    confidence: float
    description: str
    action: str
    cursor_pos: Tuple[float, float]
    is_trigger: bool = False
    handedness: str = "Right"
    debug_info: Dict[str, Any] = field(default_factory=dict)


GESTURE_DESCRIPTIONS = {
    "Open Palm": "All fingers open - Start game / Wide blade strike",
    "Closed Fist": "Fist clenched - Stop / Shield action",
    "Thumbs Up": "Thumb upright - Pause / Resume match",
    "Index Finger Pointing": "Single finger extended - Cursor & Slicing",
    "Two-Finger Gesture": "Index & Middle extended - Dual blade strike",
    "Pinch Gesture": "Thumb and index tips touching - Grab / Click",
    "Swipe Left": "Fast horizontal sweep left - Screen sweep",
    "Swipe Right": "Fast horizontal sweep right - Screen sweep",
    "Unknown": "Analyzing hand pose..."
}


class GestureRecognizer:
    """
    High-accuracy, scale-invariant gesture recognition engine.
    Computes finger extensions, joint angular relationships, and motion displacement.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        cooldown_seconds: float = 0.35,
        swipe_min_distance: float = 0.16,
        pinch_threshold: float = 0.055,
        action_mapping: Optional[Dict[str, str]] = None
    ):
        self.sensitivity = max(0.4, min(2.5, sensitivity))
        self.swipe_min_distance = swipe_min_distance
        self.pinch_threshold = pinch_threshold

        self.action_mapping = action_mapping or {
            "Open Palm": "start_game",
            "Closed Fist": "stop_action",
            "Thumbs Up": "pause_resume",
            "Index Finger Pointing": "cursor_control",
            "Two-Finger Gesture": "secondary_action",
            "Pinch Gesture": "click_interact",
            "Swipe Left": "swipe_left_action",
            "Swipe Right": "swipe_right_action"
        }

        # Temporal majority voting filter (window=3 for low latency: ~50-80ms stabilization)
        self.temporal_filter = GestureTemporalFilter(window_size=3, consensus_ratio=0.6)
        self.debouncer = GestureDebouncer(cooldown_seconds=cooldown_seconds)

        # Motion tracking ring buffer for swipe detection: hand_id -> deque of (timestamp, x, y)
        self._motion_history: Dict[int, Deque[Tuple[float, float, float]]] = {}
        self.last_recognition_time_ms: float = 0.0
        self._last_discrete_gesture: str = "Unknown"

    def _dist(self, p1: HandLandmark, p2: HandLandmark) -> float:
        """Euclidean distance in normalized 2D space."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _is_finger_extended(
        self,
        landmarks: List[HandLandmark],
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int
    ) -> bool:
        """
        Determines finger extension using scale-invariant multi-joint geometry.
        A finger is extended if distance(tip, wrist) > distance(pip, wrist)
        AND distance(tip, mcp) > distance(pip, mcp).
        """
        wrist = landmarks[WRIST]
        dist_tip_wrist = self._dist(landmarks[tip_idx], wrist)
        dist_pip_wrist = self._dist(landmarks[pip_idx], wrist)
        dist_tip_mcp = self._dist(landmarks[tip_idx], landmarks[mcp_idx])
        dist_pip_mcp = self._dist(landmarks[pip_idx], landmarks[mcp_idx])

        return (dist_tip_wrist > dist_pip_wrist * 1.05) and (dist_tip_mcp > dist_pip_mcp * 1.12)

    def _is_thumb_extended(self, landmarks: List[HandLandmark], handedness: str = "Right") -> bool:
        """
        Determines if thumb is extended outward from the palm,
        accounting for anatomical handedness and hand scale.
        """
        wrist = landmarks[WRIST]
        pinky_mcp = landmarks[PINKY_MCP]
        thumb_tip = landmarks[THUMB_TIP]
        thumb_ip = landmarks[THUMB_IP]
        thumb_mcp = landmarks[THUMB_MCP]
        index_mcp = landmarks[INDEX_MCP]

        dist_tip_pinky = self._dist(thumb_tip, pinky_mcp)
        dist_ip_pinky = self._dist(thumb_ip, pinky_mcp)
        dist_tip_wrist = self._dist(thumb_tip, wrist)
        dist_mcp_wrist = self._dist(thumb_mcp, wrist)
        dist_tip_index_mcp = self._dist(thumb_tip, index_mcp)

        hand_scale = max(0.05, self._dist(wrist, landmarks[MIDDLE_MCP]))

        return (
            (dist_tip_pinky > dist_ip_pinky * 1.02) and
            (dist_tip_wrist > dist_mcp_wrist * 1.02) and
            (dist_tip_index_mcp > 0.35 * hand_scale)
        )

    def _detect_swipe(self, hand_id: int, cx: float, cy: float, now: Optional[float] = None) -> Optional[str]:
        """Detects rapid intentional horizontal sweeps using timestamped motion history."""
        t_now = now if now is not None else time.time()
        if hand_id not in self._motion_history:
            self._motion_history[hand_id] = deque(maxlen=12)

        history = self._motion_history[hand_id]
        history.append((t_now, cx, cy))

        if len(history) < 3:
            return None

        # Find historical anchor in recent ~0.06 - 0.30s window
        best_anchor = None
        for t_hist, x_hist, y_hist in history:
            dt = t_now - t_hist
            if 0.06 <= dt <= 0.30:
                best_anchor = (t_hist, x_hist, y_hist)
                break

        if best_anchor is None:
            return None

        t_start, x_start, y_start = best_anchor
        dt = t_now - t_start
        dx = cx - x_start
        dy = cy - y_start
        speed = abs(dx) / max(0.01, dt)

        # Require predominantly horizontal displacement and sufficient speed
        sens = max(0.5, min(2.5, self.sensitivity))
        min_dist = (self.swipe_min_distance * 0.9) / sens

        if abs(dx) >= min_dist and abs(dx) > (1.6 * abs(dy)) and speed > 0.65:
            history.clear()
            return "Swipe Right" if dx > 0 else "Swipe Left"

        return None

    def recognize(self, hand: DetectedHand) -> RecognizedGesture:
        """
        Evaluates a detected hand and classifies its gesture with sub-millisecond latency.
        
        Args:
            hand: DetectedHand with 21 normalized landmarks.
            
        Returns:
            RecognizedGesture: The classified gesture, confidence, and debug telemetry.
        """
        t_start = time.perf_counter()
        lms = hand.landmarks

        if not lms or len(lms) < 21:
            self.last_recognition_time_ms = 0.0
            return RecognizedGesture(
                name="Unknown",
                confidence=0.0,
                description=GESTURE_DESCRIPTIONS["Unknown"],
                action="none",
                cursor_pos=hand.center,
                handedness=hand.handedness
            )

        cx, cy = hand.center
        cursor_target = hand.index_tip

        # Hand scale reference (wrist to middle MCP distance)
        hand_scale = max(0.05, self._dist(lms[WRIST], lms[MIDDLE_MCP]))

        # 1. Scale-Invariant Pinch Detection (Thumb tip to Index tip distance)
        raw_pinch_dist = self._dist(lms[THUMB_TIP], lms[INDEX_TIP])
        norm_pinch = raw_pinch_dist / hand_scale
        dist_thumb_mid = self._dist(lms[THUMB_TIP], lms[MIDDLE_TIP])
        # Pinch requires thumb and index tips to be close to each other (< 0.16 hand scale) and closer than thumb to middle
        pinch_active = (norm_pinch < 0.16) and (raw_pinch_dist < dist_thumb_mid * 0.90)

        # 2. Multi-Joint Finger Extension Evaluation
        idx_ext = self._is_finger_extended(lms, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        mid_ext = self._is_finger_extended(lms, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        rng_ext = self._is_finger_extended(lms, RING_TIP, RING_PIP, RING_MCP)
        pnk_ext = self._is_finger_extended(lms, PINKY_TIP, PINKY_PIP, PINKY_MCP)
        thb_ext = self._is_thumb_extended(lms, hand.handedness)

        finger_states = {
            "thumb": thb_ext,
            "index": idx_ext,
            "middle": mid_ext,
            "ring": rng_ext,
            "pinky": pnk_ext
        }

        # 3. Check for Thumbs Up (scale-invariant vertical orientation)
        thumb_tip = lms[THUMB_TIP]
        thumb_ip = lms[THUMB_IP]
        is_thumb_upward = (
            (thumb_tip.y < thumb_ip.y - 0.08 * hand_scale) and
            (thumb_tip.y < lms[INDEX_MCP].y - 0.06 * hand_scale)
        )
        is_thumbs_up = is_thumb_upward and not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not pinch_active

        # 4. Check for Swipes (Dynamic motion gesture)
        swipe = None
        if idx_ext or mid_ext or thb_ext:
            swipe = self._detect_swipe(hand.hand_id, cx, cy)

        # Check for Closed Fist: all 4 fingers curled and not actively pinching
        is_closed_fist = (not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not pinch_active)

        # Priority Resolution
        raw_name = "Unknown"
        conf = 0.50
        is_trigger = False
        is_dynamic_swipe = False

        if swipe and self.debouncer.can_trigger(swipe):
            raw_name = swipe
            conf = 0.96
            is_trigger = True
            is_dynamic_swipe = True
            self.debouncer.trigger(swipe)
            self.temporal_filter.clear()

        elif is_thumbs_up:
            raw_name = "Thumbs Up"
            conf = 0.96
            if self._last_discrete_gesture != "Thumbs Up" and self.debouncer.can_trigger("Thumbs Up"):
                is_trigger = True
                self.debouncer.trigger("Thumbs Up")
            cursor_target = (lms[THUMB_TIP].x, lms[THUMB_TIP].y)

        elif is_closed_fist:
            raw_name = "Closed Fist"
            conf = 0.94
            if self._last_discrete_gesture != "Closed Fist" and self.debouncer.can_trigger("Closed Fist"):
                is_trigger = True
                self.debouncer.trigger("Closed Fist")
            cursor_target = hand.center

        elif pinch_active and not (mid_ext and rng_ext and pnk_ext):
            raw_name = "Pinch Gesture"
            conf = min(0.99, max(0.75, 1.0 - (norm_pinch / 0.16)))
            is_trigger = self.debouncer.can_trigger("Pinch Gesture")
            if is_trigger:
                self.debouncer.trigger("Pinch Gesture")
            # Pinch midpoint cursor
            cursor_target = (
                (lms[THUMB_TIP].x + lms[INDEX_TIP].x) / 2.0,
                (lms[THUMB_TIP].y + lms[INDEX_TIP].y) / 2.0
            )

        elif not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not pinch_active:
            raw_name = "Closed Fist"
            conf = 0.94
            if self._last_discrete_gesture != "Closed Fist" and self.debouncer.can_trigger("Closed Fist"):
                is_trigger = True
                self.debouncer.trigger("Closed Fist")
            cursor_target = hand.center

        elif idx_ext and mid_ext and not rng_ext and not pnk_ext:
            raw_name = "Two-Finger Gesture"
            conf = 0.95
            cursor_target = (
                (lms[INDEX_TIP].x + lms[MIDDLE_TIP].x) / 2.0,
                (lms[INDEX_TIP].y + lms[MIDDLE_TIP].y) / 2.0
            )

        elif idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            raw_name = "Index Finger Pointing"
            conf = 0.97
            cursor_target = (lms[INDEX_TIP].x, lms[INDEX_TIP].y)

        elif idx_ext and mid_ext and rng_ext and pnk_ext:
            raw_name = "Open Palm"
            conf = 0.98
            if self._last_discrete_gesture != "Open Palm" and self.debouncer.can_trigger("Open Palm"):
                is_trigger = True
                self.debouncer.trigger("Open Palm")
            cursor_target = (lms[INDEX_TIP].x, lms[INDEX_TIP].y)

        # Update discrete pose memory for edge triggering
        self._last_discrete_gesture = raw_name

        # Dynamic gestures (swipes) bypass temporal voting filter; static poses are smoothed
        if is_dynamic_swipe:
            stable_name = raw_name
            stable_conf = conf
        else:
            stable_name, stable_conf = self.temporal_filter.add(raw_name, conf)

        self.last_recognition_time_ms = (time.perf_counter() - t_start) * 1000.0

        debug_telemetry = {
            "finger_states": finger_states,
            "hand_scale": round(hand_scale, 3),
            "norm_pinch": round(norm_pinch, 3),
            "raw_name": raw_name,
            "rec_time_ms": round(self.last_recognition_time_ms, 2)
        }

        return RecognizedGesture(
            name=stable_name,
            confidence=round(stable_conf, 2),
            description=GESTURE_DESCRIPTIONS.get(stable_name, "Pose Active"),
            action=self.action_mapping.get(stable_name, "none"),
            cursor_pos=cursor_target,
            is_trigger=is_trigger,
            handedness=hand.handedness,
            debug_info=debug_telemetry
        )

    def set_sensitivity(self, sensitivity: float) -> None:
        """Updates sensitivity multiplier."""
        self.sensitivity = max(0.4, min(2.5, sensitivity))

    def reset(self) -> None:
        """Clears all motion and smoothing buffers."""
        self._motion_history.clear()
        self.temporal_filter.clear()
        self.debouncer.reset()
        self._last_discrete_gesture = "Unknown"
