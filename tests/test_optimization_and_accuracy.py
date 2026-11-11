"""
Comprehensive Unit Tests for Optimization and Hand Tracking Accuracy.
Covers:
- Coordinate mapping with sensitivity multipliers and mirror toggling
- High-speed cursor-trail continuous interpolation and multi-mode slicing
- Scale-invariant gesture classification and dynamic swipe bypass
- Tracking loss grace timeout and state reset
"""

import math
import time
import pytest
import numpy as np

from src.detection.hand_detector import (
    DetectedHand,
    HandLandmark,
    INDEX_TIP,
    INDEX_MCP,
    INDEX_PIP,
    MIDDLE_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    RING_TIP,
    RING_MCP,
    RING_PIP,
    PINKY_TIP,
    PINKY_MCP,
    PINKY_PIP,
    THUMB_TIP,
    THUMB_IP,
    THUMB_MCP,
    WRIST
)
from src.detection.hand_tracker import HandTracker, TrackingStatus
from src.detection.one_euro_filter import OneEuroFilter2D
from src.game.collision import check_trail_fruit_collision, line_segment_circle_intersect
from src.game.fruit import Fruit, FruitType
from src.game.game_engine import GameEngine, GameState
from src.gestures.gesture_recognizer import GestureRecognizer, RecognizedGesture


def make_scaled_hand(
    x: float = 0.5,
    y: float = 0.5,
    scale: float = 0.20,
    pose: str = "pointing",
    handedness: str = "Right"
) -> DetectedHand:
    """Creates an anatomically scaled synthetic 21-landmark hand."""
    lms = [HandLandmark(x=x, y=y, z=0.0) for _ in range(21)]
    wrist = HandLandmark(x=x, y=y + scale)
    lms[WRIST] = wrist

    # Knuckles (MCPs)
    lms[INDEX_MCP] = HandLandmark(x=x - scale * 0.15, y=y)
    lms[MIDDLE_MCP] = HandLandmark(x=x, y=y)
    lms[RING_MCP] = HandLandmark(x=x + scale * 0.15, y=y + scale * 0.05)
    lms[PINKY_MCP] = HandLandmark(x=x + scale * 0.30, y=y + scale * 0.10)
    lms[THUMB_MCP] = HandLandmark(x=x - scale * 0.30, y=y + scale * 0.35)
    lms[THUMB_IP] = HandLandmark(x=x - scale * 0.35, y=y + scale * 0.20)

    # Defaults: curled
    lms[INDEX_PIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y - scale * 0.25)
    lms[INDEX_TIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y - scale * 0.15)
    lms[MIDDLE_PIP] = HandLandmark(x=lms[MIDDLE_MCP].x, y=y - scale * 0.25)
    lms[MIDDLE_TIP] = HandLandmark(x=lms[MIDDLE_MCP].x, y=y - scale * 0.15)
    lms[RING_PIP] = HandLandmark(x=lms[RING_MCP].x, y=y - scale * 0.20)
    lms[RING_TIP] = HandLandmark(x=lms[RING_MCP].x, y=y - scale * 0.10)
    lms[PINKY_PIP] = HandLandmark(x=lms[PINKY_MCP].x, y=y - scale * 0.15)
    lms[PINKY_TIP] = HandLandmark(x=lms[PINKY_MCP].x, y=y - scale * 0.08)
    lms[THUMB_TIP] = HandLandmark(x=lms[THUMB_MCP].x, y=y + scale * 0.25)

    if pose == "pointing":
        # Index extended, others curled
        lms[INDEX_PIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y - scale * 0.35)
        lms[INDEX_TIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y - scale * 0.70)
    elif pose == "open_palm":
        # All extended
        lms[INDEX_PIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y - scale * 0.35)
        lms[INDEX_TIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y - scale * 0.70)
        lms[MIDDLE_PIP] = HandLandmark(x=lms[MIDDLE_MCP].x, y=y - scale * 0.40)
        lms[MIDDLE_TIP] = HandLandmark(x=lms[MIDDLE_MCP].x, y=y - scale * 0.80)
        lms[RING_PIP] = HandLandmark(x=lms[RING_MCP].x, y=y - scale * 0.35)
        lms[RING_TIP] = HandLandmark(x=lms[RING_MCP].x, y=y - scale * 0.70)
        lms[PINKY_PIP] = HandLandmark(x=lms[PINKY_MCP].x, y=y - scale * 0.25)
        lms[PINKY_TIP] = HandLandmark(x=lms[PINKY_MCP].x, y=y - scale * 0.55)
        lms[THUMB_TIP] = HandLandmark(x=x - scale * 0.65, y=y + scale * 0.10)
    elif pose == "pinch":
        # Thumb tip touches index tip
        pinch_pt = HandLandmark(x=x - scale * 0.20, y=y - scale * 0.20)
        lms[THUMB_TIP] = pinch_pt
        lms[INDEX_TIP] = HandLandmark(x=pinch_pt.x + 0.01 * scale, y=pinch_pt.y + 0.01 * scale)
        # Middle finger clearly separated
        lms[MIDDLE_TIP] = HandLandmark(x=x + scale * 0.20, y=y + scale * 0.10)
    elif pose == "thumbs_up":
        # Thumb points straight up
        lms[THUMB_IP] = HandLandmark(x=lms[THUMB_MCP].x, y=y - scale * 0.15)
        lms[THUMB_TIP] = HandLandmark(x=lms[THUMB_MCP].x, y=y - scale * 0.45)
    elif pose == "closed_fist":
        # All knuckles curled
        lms[INDEX_TIP] = HandLandmark(x=lms[INDEX_MCP].x, y=y + scale * 0.10)
        lms[MIDDLE_TIP] = HandLandmark(x=lms[MIDDLE_MCP].x, y=y + scale * 0.10)
        lms[RING_TIP] = HandLandmark(x=lms[RING_MCP].x, y=y + scale * 0.10)
        lms[PINKY_TIP] = HandLandmark(x=lms[PINKY_MCP].x, y=y + scale * 0.10)
        lms[THUMB_TIP] = HandLandmark(x=lms[THUMB_MCP].x, y=y + scale * 0.25)

    return DetectedHand(
        landmarks=lms,
        bbox=(int((x - 0.1) * 640), int((y - 0.1) * 480), int(scale * 640), int(scale * 480)),
        handedness=handedness,
        confidence=0.98,
        hand_id=0
    )


class TestCoordinateMappingAndSensitivity:

    def test_mirrored_vs_unmirrored_mapping(self):
        """Unmirrored feeds must horizontally invert x to preserve natural physical direction."""
        tracker_mirrored = HandTracker(sensitivity=1.0, is_mirrored=True)
        tracker_unmirrored = HandTracker(sensitivity=1.0, is_mirrored=False)

        # Hand on right of frame (x = 0.70)
        hand = make_scaled_hand(x=0.70, y=0.50)

        _, _, pos_m = tracker_mirrored.update([hand], now=1.0)
        _, _, pos_u = tracker_unmirrored.update([hand], now=1.0)

        assert abs(pos_m[0] - 0.70) < 0.05
        # In unmirrored feed, x=0.70 maps to 1.0 - 0.70 = 0.30
        assert abs(pos_u[0] - 0.30) < 0.05

    def test_sensitivity_multiplier_expansion(self):
        """Sensitivity > 1.0 expands coordinate reach around center (0.5)."""
        tracker_norm = HandTracker(sensitivity=1.0, is_mirrored=True)
        tracker_sens = HandTracker(sensitivity=1.5, is_mirrored=True)

        hand_off_center = make_scaled_hand(x=0.60, y=0.50)  # +0.10 offset from center

        _, _, pos_norm = tracker_norm.update([hand_off_center], now=1.0)
        _, _, pos_sens = tracker_sens.update([hand_off_center], now=1.0)

        offset_norm = pos_norm[0] - 0.5
        offset_sens = pos_sens[0] - 0.5
        # Sensitivity 1.5 must produce 1.5x displacement
        assert abs(offset_sens - offset_norm * 1.5) < 1e-3

    def test_coordinate_clamping(self):
        """Sensitivity must never output coordinates outside [0.0, 1.0]."""
        tracker = HandTracker(sensitivity=2.5, is_mirrored=True)
        hand_edge = make_scaled_hand(x=0.95, y=0.95)

        _, _, pos = tracker.update([hand_edge], now=1.0)
        assert 0.0 <= pos[0] <= 1.0
        assert 0.0 <= pos[1] <= 1.0


class TestContinuousCollisionAndSlicingModes:

    def test_high_speed_subsegment_interpolation_slices_fruit(self):
        """Rapid movement across 120 pixels in 1 frame must interpolate and slice target."""
        engine = GameEngine(width=1280, height=720, sound_enabled=False)
        engine.state = GameState.PLAYING

        fruit = Fruit(x=500.0, y=360.0, vx=0.0, vy=0.0, fruit_data=FruitType.WATERMELON)
        engine.fruits.append(fruit)

        # Fast swing across fruit: start at x=420, end at x=580 (jump of 160 px in one frame)
        engine.update_cursor((420.0 / 1280.0, 360.0 / 720.0), slicing_mode="motion")
        engine.update_cursor((580.0 / 1280.0, 360.0 / 720.0), slicing_mode="motion")

        assert fruit.is_sliced is True
        assert engine.fruits_sliced == 1

    def test_slicing_mode_pinch_activation(self):
        """Pinch slicing mode only slices when pinch gesture is active."""
        engine = GameEngine(width=1280, height=720, sound_enabled=False)
        engine.state = GameState.PLAYING

        fruit = Fruit(x=500.0, y=360.0, vx=0.0, vy=0.0, fruit_data=FruitType.APPLE)
        engine.fruits.append(fruit)

        # 1. Moving across fruit without pinch -> NOT sliced
        engine.update_cursor((450.0 / 1280.0, 360.0 / 720.0), active_slice=False, slicing_mode="pinch")
        engine.update_cursor((550.0 / 1280.0, 360.0 / 720.0), active_slice=False, slicing_mode="pinch")
        assert fruit.is_sliced is False

        # 2. Moving across fruit WITH pinch -> SLICED
        engine.update_cursor((450.0 / 1280.0, 360.0 / 720.0), active_slice=True, slicing_mode="pinch")
        engine.update_cursor((550.0 / 1280.0, 360.0 / 720.0), active_slice=True, slicing_mode="pinch")
        assert fruit.is_sliced is True

    def test_stale_cursor_cleared_on_hand_loss(self):
        """Cursor trail must immediately clear when hand tracking is lost."""
        engine = GameEngine(width=1280, height=720, sound_enabled=False)
        engine.update_cursor((0.5, 0.5))
        engine.update_cursor((0.55, 0.55))
        assert len(engine.cursor_trail) > 0

        engine.clear_cursor()
        assert len(engine.cursor_trail) == 0


class TestGestureAccuracyAndDynamicSwipe:

    def test_closed_fist_vs_pinch_discrimination(self):
        """Closed fist must NOT trigger Pinch Gesture false positive."""
        recognizer = GestureRecognizer()
        fist_hand = make_scaled_hand(pose="closed_fist")

        # Evaluate across 3 frames to establish stability
        gest = None
        for _ in range(3):
            gest = recognizer.recognize(fist_hand)

        assert gest is not None
        assert gest.name == "Closed Fist"
        assert gest.name != "Pinch Gesture"

    def test_pinch_scale_invariance_small_and_large_hands(self):
        """Pinch gesture triggers consistently for both small (far) and large (close) hands."""
        recognizer = GestureRecognizer()

        small_pinch = make_scaled_hand(scale=0.12, pose="pinch")
        large_pinch = make_scaled_hand(scale=0.35, pose="pinch")

        for _ in range(3):
            res_small = recognizer.recognize(small_pinch)
        for _ in range(3):
            res_large = recognizer.recognize(large_pinch)

        assert res_small.name == "Pinch Gesture"
        assert res_large.name == "Pinch Gesture"

    def test_dynamic_swipe_bypasses_temporal_filter(self):
        """Dynamic swipes trigger immediately and work in both directions."""
        recognizer = GestureRecognizer(swipe_min_distance=0.15)
        hand_id = 1

        # Simulate fast left-to-right swipe (Swipe Right)
        t = 100.0
        # Start at 0.30
        recognizer._motion_history[hand_id] = [
            (t, 0.30, 0.50),
            (t + 0.05, 0.40, 0.50),
            (t + 0.10, 0.55, 0.50)
        ]
        # Current frame at t+0.12, x=0.60
        swipe = recognizer._detect_swipe(hand_id, cx=0.60, cy=0.50, now=t + 0.12)
        assert swipe == "Swipe Right"

        # Simulate fast right-to-left swipe (Swipe Left)
        recognizer.reset()
        recognizer._motion_history[hand_id] = [
            (t, 0.70, 0.50),
            (t + 0.05, 0.55, 0.50),
            (t + 0.10, 0.45, 0.50)
        ]
        swipe_left = recognizer._detect_swipe(hand_id, cx=0.35, cy=0.50, now=t + 0.12)
        assert swipe_left == "Swipe Left"

    def test_fast_motion_landmark_continuity_and_extrapolation(self):
        """Tracker predicts position and preserves landmarks during brief high-speed detection dropouts."""
        tracker = HandTracker(tracking_loss_timeout=0.25, sensitivity=1.0)
        
        # Frame 1: Hand moving rapidly right
        h1 = make_scaled_hand(x=0.30, y=0.50)
        status1, locked1, pos1 = tracker.update([h1], now=10.0)
        assert status1 == TrackingStatus.TRACKING
        
        # Frame 2: Hand at 0.40 (vx = 0.10 / 0.033 = 3.03 units/sec)
        h2 = make_scaled_hand(x=0.40, y=0.50)
        status2, locked2, pos2 = tracker.update([h2], now=10.033)
        assert status2 == TrackingStatus.TRACKING
        
        orig_index_x = h2.landmarks[INDEX_TIP].x
        # Frame 3: Rapid motion blur causes detection dropout for 1 frame
        status3, locked3, pos3 = tracker.update([], now=10.066)
        assert status3 == TrackingStatus.SEARCHING
        assert locked3 is not None
        assert len(locked3.landmarks) == 21
        # Position must have moved forward due to velocity extrapolation rather than freezing or jumping backwards
        assert pos3[0] > pos2[0]
        # Landmarks must have shifted forward continuously
        assert locked3.landmarks[INDEX_TIP].x > orig_index_x
