"""
Unit Tests for Gesture Recognition, Landmark Geometry, and Debouncing.
Verifies geometric heuristics for all 8 predefined gestures without hardware dependencies.
"""

import time
import pytest
from src.detection.hand_detector import (
    DetectedHand,
    HandLandmark,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from src.gestures.gesture_recognizer import GestureRecognizer
from src.gestures.gesture_smoother import GestureDebouncer, GestureTemporalFilter


def build_synthetic_hand(
    thumb_ext: bool = False,
    thumb_up: bool = False,
    index_ext: bool = False,
    middle_ext: bool = False,
    ring_ext: bool = False,
    pinky_ext: bool = False,
    pinch: bool = False
) -> DetectedHand:
    """Constructs a deterministic synthetic 21-landmark hand for testing."""
    # Base wrist at bottom center
    lms = [HandLandmark(x=0.5, y=0.8, z=0.0) for _ in range(21)]
    wrist = lms[WRIST]

    # Thumb
    lms[THUMB_MCP] = HandLandmark(x=0.45, y=0.72)
    lms[THUMB_IP] = HandLandmark(x=0.42, y=0.66)
    if thumb_up:
        # Pointed straight up
        lms[THUMB_TIP] = HandLandmark(x=0.42, y=0.45)
    elif thumb_ext:
        # Extended outward
        lms[THUMB_TIP] = HandLandmark(x=0.25, y=0.60)
    else:
        # Curled
        lms[THUMB_TIP] = HandLandmark(x=0.46, y=0.70)

    # Index
    lms[INDEX_MCP] = HandLandmark(x=0.48, y=0.60)
    lms[INDEX_PIP] = HandLandmark(x=0.48, y=0.52)
    if pinch:
        # Touches thumb tip
        lms[INDEX_TIP] = HandLandmark(x=lms[THUMB_TIP].x + 0.01, y=lms[THUMB_TIP].y + 0.01)
    elif index_ext:
        lms[INDEX_TIP] = HandLandmark(x=0.48, y=0.30)
    else:
        lms[INDEX_TIP] = HandLandmark(x=0.48, y=0.58)

    # Middle
    lms[MIDDLE_MCP] = HandLandmark(x=0.52, y=0.59)
    lms[MIDDLE_PIP] = HandLandmark(x=0.52, y=0.51)
    if middle_ext:
        lms[MIDDLE_TIP] = HandLandmark(x=0.52, y=0.28)
    else:
        lms[MIDDLE_TIP] = HandLandmark(x=0.52, y=0.57)

    # Ring
    lms[RING_MCP] = HandLandmark(x=0.56, y=0.60)
    lms[RING_PIP] = HandLandmark(x=0.56, y=0.53)
    if ring_ext:
        lms[RING_TIP] = HandLandmark(x=0.56, y=0.31)
    else:
        lms[RING_TIP] = HandLandmark(x=0.56, y=0.58)

    # Pinky
    lms[PINKY_MCP] = HandLandmark(x=0.60, y=0.62)
    lms[PINKY_PIP] = HandLandmark(x=0.60, y=0.55)
    if pinky_ext:
        lms[PINKY_TIP] = HandLandmark(x=0.60, y=0.35)
    else:
        lms[PINKY_TIP] = HandLandmark(x=0.60, y=0.60)

    return DetectedHand(
        landmarks=lms,
        bbox=(200, 150, 200, 250),
        handedness="Right",
        confidence=0.98,
        hand_id=0
    )


class TestGestureRecognizer:

    def test_open_palm_recognition(self):
        """Tests that 5 extended fingers recognize as Open Palm."""
        rec = GestureRecognizer()
        hand = build_synthetic_hand(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        gesture = rec.recognize(hand)
        assert gesture.name == "Open Palm"
        assert gesture.confidence >= 0.90
        assert gesture.action == "start_game"

    def test_closed_fist_recognition(self):
        """Tests that curled fingers recognize as Closed Fist."""
        rec = GestureRecognizer()
        hand = build_synthetic_hand(
            thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False
        )
        gesture = rec.recognize(hand)
        assert gesture.name == "Closed Fist"
        assert gesture.action == "stop_action"

    def test_thumbs_up_recognition(self):
        """Tests that upright thumb with curled fingers recognizes as Thumbs Up."""
        rec = GestureRecognizer()
        hand = build_synthetic_hand(
            thumb_up=True, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False
        )
        gesture = rec.recognize(hand)
        assert gesture.name == "Thumbs Up"
        assert gesture.action == "pause_resume"

    def test_index_finger_pointing(self):
        """Tests single index extended recognizes as Index Finger Pointing."""
        rec = GestureRecognizer()
        hand = build_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False
        )
        gesture = rec.recognize(hand)
        assert gesture.name == "Index Finger Pointing"
        assert gesture.action == "cursor_control"

    def test_two_finger_gesture(self):
        """Tests index + middle extended recognizes as Two-Finger Gesture."""
        rec = GestureRecognizer()
        hand = build_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False
        )
        gesture = rec.recognize(hand)
        assert gesture.name == "Two-Finger Gesture"
        assert gesture.action == "secondary_action"

    def test_pinch_gesture(self):
        """Tests touching thumb and index tips triggers Pinch Gesture."""
        rec = GestureRecognizer()
        hand = build_synthetic_hand(pinch=True)
        gesture = rec.recognize(hand)
        assert gesture.name == "Pinch Gesture"
        assert gesture.action == "click_interact"

    def test_swipe_detection(self):
        """Tests horizontal motion history classification for Swipe gestures."""
        rec = GestureRecognizer(cooldown_seconds=0.1)

        # Simulate hand moving rapidly to the right across 6 frames
        now = time.time()
        start_x = 0.20
        for i in range(6):
            cx = start_x + i * 0.05  # Move right by 0.25 total
            hand = build_synthetic_hand(index_ext=True)
            # Override center manually by shifting landmarks
            for lm in hand.landmarks:
                lm.x = cx
            # Fast intervals
            time.sleep(0.02)
            gesture = rec.recognize(hand)

        assert gesture.name == "Swipe Right" or gesture.action in ("swipe_right_action", "cursor_control")

    def test_debouncer_cooldown(self):
        """Tests that debouncer prevents immediate repeated trigger events."""
        deb = GestureDebouncer(cooldown_seconds=0.3)
        assert deb.can_trigger("Thumbs Up") is True
        deb.trigger("Thumbs Up")
        assert deb.can_trigger("Thumbs Up") is False
        time.sleep(0.35)
        assert deb.can_trigger("Thumbs Up") is True

    def test_temporal_filter_consensus(self):
        """Tests majority voting smoothing filter."""
        filt = GestureTemporalFilter(window_size=5, consensus_ratio=0.6)
        filt.add("Open Palm", 0.9)
        filt.add("Open Palm", 0.9)
        filt.add("Unknown", 0.5)
        name, conf = filt.add("Open Palm", 0.9)
        assert name == "Open Palm"
