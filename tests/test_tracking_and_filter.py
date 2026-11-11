"""
Unit Tests for One Euro Adaptive Filtering, Hand Tracking Continuity, and Continuous Slicing.
Validates latency reduction, jitter suppression, target lock continuity, and collision interpolation.
"""

import math
import time
import numpy as np
import pytest

from src.detection.one_euro_filter import OneEuroFilter, OneEuroFilter2D
from src.detection.hand_tracker import HandTracker, TrackingStatus
from src.detection.hand_detector import DetectedHand, HandLandmark, WRIST, MIDDLE_MCP, INDEX_TIP, THUMB_TIP
from src.game.game_engine import GameEngine, GameState
from src.game.fruit import Fruit, FruitType
from src.gestures.gesture_recognizer import GestureRecognizer


def create_mock_hand(
    x: float = 0.5,
    y: float = 0.5,
    handedness: str = "Right",
    confidence: float = 0.95,
    hand_id: int = 0,
    scale: float = 0.25
) -> DetectedHand:
    """Helper to generate a mock DetectedHand with specified position and scale."""
    landmarks = [HandLandmark(x=x, y=y, z=0.0) for _ in range(21)]
    landmarks[WRIST] = HandLandmark(x=x, y=y + scale)
    landmarks[MIDDLE_MCP] = HandLandmark(x=x, y=y)
    landmarks[INDEX_TIP] = HandLandmark(x=x, y=y - scale * 0.5)
    landmarks[THUMB_TIP] = HandLandmark(x=x - scale * 0.3, y=y)

    return DetectedHand(
        landmarks=landmarks,
        bbox=(int((x - 0.1) * 640), int((y - 0.1) * 480), 100, 100),
        handedness=handedness,
        confidence=confidence,
        hand_id=hand_id
    )


class TestOneEuroFilter:

    def test_jitter_suppression_at_stationary(self):
        """At near-zero velocity, high-frequency jitter should be strongly attenuated."""
        f = OneEuroFilter(min_cutoff=1.0, beta=15.0)
        base_value = 0.50
        np.random.seed(42)

        raw_samples = []
        filtered_samples = []
        t = 0.0
        dt = 0.033  # ~30 FPS

        # Feed 60 noisy samples around 0.50
        for _ in range(60):
            noise = np.random.normal(0.0, 0.02)  # +/- 2% sensor noise
            val = base_value + noise
            raw_samples.append(val)
            filtered = f.filter(val, timestamp=t)
            filtered_samples.append(filtered)
            t += dt

        # Skip warm-up first 5 samples
        raw_std = float(np.std(raw_samples[5:]))
        filtered_std = float(np.std(filtered_samples[5:]))

        # Filtered output variance must be significantly smaller than raw noise
        assert filtered_std < raw_std * 0.65

    def test_low_latency_during_rapid_movement(self):
        """At high velocity, dynamic cutoff expands, minimizing phase lag."""
        f = OneEuroFilter(min_cutoff=1.0, beta=15.0)
        t = 0.0
        dt = 0.016  # 60 FPS

        # Settle at 0.0
        for _ in range(20):
            f.filter(0.0, timestamp=t)
            t += dt

        # High-speed ramp from 0.0 to 1.0 in ~100ms
        ramp_samples = [0.2, 0.4, 0.6, 0.8, 1.0]
        final_filtered = 0.0
        for sample in ramp_samples:
            final_filtered = f.filter(sample, timestamp=t)
            t += dt

        # Because beta=15 expands cutoff during fast movement, final value tracks closely (> 0.85)
        assert final_filtered > 0.85

    def test_filter_reset(self):
        """Reset clears previous history so new signals start immediately."""
        f = OneEuroFilter()
        f.filter(100.0, timestamp=1.0)
        f.filter(100.0, timestamp=1.033)

        f.reset()
        assert f.x_filt.s is None
        assert f.last_time is None

        # After reset, first sample is returned directly without smoothing lag
        out = f.filter(10.0, timestamp=2.0)
        assert abs(out - 10.0) < 1e-4

    def test_2d_filter_independence(self):
        """OneEuroFilter2D filters x and y channels simultaneously."""
        f2d = OneEuroFilter2D(min_cutoff=1.2, beta=15.0)
        rx, ry = f2d.filter(0.3, 0.7, timestamp=0.0)
        assert abs(rx - 0.3) < 1e-4
        assert abs(ry - 0.7) < 1e-4

        rx2, ry2 = f2d.filter(0.35, 0.75, timestamp=0.033)
        assert 0.3 < rx2 <= 0.35
        assert 0.7 < ry2 <= 0.75


class TestHandTracker:

    def test_single_hand_tracking_acquisition(self):
        """HandTracker should acquire single hand and enter TRACKING status."""
        tracker = HandTracker(preferred_handedness="Right", control_point="index_tip")
        hand = create_mock_hand(x=0.4, y=0.4, handedness="Right")

        status, locked_hand, pos = tracker.update([hand], now=1.0)
        assert status == TrackingStatus.TRACKING
        assert locked_hand is hand
        assert pos is not None
        assert 0.0 <= pos[0] <= 1.0
        assert 0.0 <= pos[1] <= 1.0

    def test_hand_identity_continuity_multiple_hands(self):
        """When multiple hands appear, tracker locks onto the spatially closest hand."""
        tracker = HandTracker(preferred_handedness="Right")
        hand_a1 = create_mock_hand(x=0.3, y=0.5, handedness="Right", hand_id=0)
        tracker.update([hand_a1], now=1.0)

        # In next frame, two hands appear: Hand A slightly moved to 0.32, Hand B at 0.80
        hand_a2 = create_mock_hand(x=0.32, y=0.51, handedness="Right", hand_id=0)
        hand_b = create_mock_hand(x=0.80, y=0.50, handedness="Left", hand_id=1)

        status, locked, pos = tracker.update([hand_b, hand_a2], now=1.033)
        assert status == TrackingStatus.TRACKING
        assert locked is hand_a2  # Retains hand_a due to spatial proximity

    def test_tracking_loss_grace_timeout(self):
        """Brief hand loss enters SEARCHING status and preserves last position."""
        tracker = HandTracker(tracking_loss_timeout=0.15)
        hand = create_mock_hand(x=0.5, y=0.5)
        _, _, initial_pos = tracker.update([hand], now=1.0)

        # 50ms later, hand detection drops
        status, preserved_hand, preserved_pos = tracker.update([], now=1.05)
        assert status == TrackingStatus.SEARCHING
        assert preserved_pos == initial_pos

    def test_tracking_loss_expiration(self):
        """Hand loss exceeding timeout transitions to LOST and resets position."""
        tracker = HandTracker(tracking_loss_timeout=0.15)
        hand = create_mock_hand(x=0.5, y=0.5)
        tracker.update([hand], now=1.0)

        # 200ms later (exceeds 150ms timeout)
        status, locked, pos = tracker.update([], now=1.20)
        assert status == TrackingStatus.LOST
        assert locked is None
        assert pos is None


class TestGameEngineContinuousCollision:

    def test_sub_segment_line_interpolation_slices_fast_fruit(self):
        """Fast hand leap between two frames must hit fruit via line interpolation."""
        engine = GameEngine()
        engine.state = GameState.PLAYING
        engine.fruits.clear()

        # Place stationary fruit directly in the path at (300, 300)
        fruit = Fruit(x=300.0, y=300.0, vx=0.0, vy=0.0, fruit_data=FruitType.WATERMELON)
        fruit.radius = 35.0
        engine.fruits.append(fruit)

        w, h = engine.width, engine.height
        # First cursor point far left: (100, 300)
        engine.update_cursor((100.0 / w, 300.0 / h), active_slice=True)

        # Fast sweep to far right: (500, 300) in a single tick (400px jump across fruit)
        engine.update_cursor((500.0 / w, 300.0 / h), active_slice=True)

        # The fruit must be sliced because sub-segment interpolation traversed (300, 300)
        assert fruit.is_sliced is True

    def test_clear_cursor_on_tracking_loss(self):
        """clear_cursor removes active trail and prevents stale slicing."""
        engine = GameEngine()
        engine.update_cursor((0.5, 0.5), active_slice=True)
        assert len(engine.cursor_trail) > 0

        engine.clear_cursor()
        assert len(engine.cursor_trail) == 0


class TestScaleInvariantPinch:

    def test_pinch_scale_invariance(self):
        """Pinch detection works reliably across both large and small hand scales."""
        recognizer = GestureRecognizer()

        for scale in [0.12, 0.25, 0.40]:  # Child hand, normal hand, large hand
            landmarks = [HandLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            landmarks[WRIST] = HandLandmark(x=0.5, y=0.5 + scale)
            landmarks[MIDDLE_MCP] = HandLandmark(x=0.5, y=0.5)

            # Thumb and Index tips touching
            landmarks[THUMB_TIP] = HandLandmark(x=0.45, y=0.42)
            landmarks[INDEX_TIP] = HandLandmark(x=0.45 + (scale * 0.05), y=0.42 + (scale * 0.05))

            hand = DetectedHand(
                landmarks=landmarks,
                bbox=(100, 100, 200, 200),
                handedness="Right",
                confidence=0.98,
                hand_id=0
            )

            result = recognizer.recognize(hand)
            assert result.name == "Pinch Gesture", f"Failed for hand scale {scale}"
