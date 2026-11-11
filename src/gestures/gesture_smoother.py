"""
Gesture Temporal Smoothing & Debouncing Module.
Stabilizes raw frame-by-frame gesture classifications using temporal consensus
and enforces configurable cooldown periods for trigger-style actions.
"""

import time
from collections import Counter, deque
from typing import Deque, Dict, Optional, Tuple


class GestureDebouncer:
    """
    Enforces a cooldown period to prevent rapid repeated triggering
    of the same gesture event (e.g. Pause, Swipes, Button Clicks).
    """

    def __init__(self, cooldown_seconds: float = 0.35):
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger_times: Dict[str, float] = {}

    def can_trigger(self, gesture_name: str) -> bool:   
        """Checks if enough time has elapsed since this gesture last fired."""
        now = time.time()
        last_time = self._last_trigger_times.get(gesture_name, 0.0)
        return (now - last_time) >= self.cooldown_seconds

    def trigger(self, gesture_name: str) -> None:
        """Records a successful trigger timestamp for the gesture."""
        self._last_trigger_times[gesture_name] = time.time()

    def reset(self) -> None:
        """Clears all cooldown timestamps."""
        self._last_trigger_times.clear()


class GestureTemporalFilter:
    """
    Temporal majority voting filter over a sliding window of recent predictions
    to eliminate frame-to-frame noise and flicker.
    """

    def __init__(self, window_size: int = 5, consensus_ratio: float = 0.6):
        self.window_size = window_size
        self.consensus_ratio = consensus_ratio
        self._history: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def add(self, gesture_name: str, confidence: float) -> Tuple[str, float]:
        """
        Adds a new frame prediction and returns the stabilized consensus gesture.
        
        Returns:
            Tuple[str, float]: (Consensus gesture name, average confidence)
        """
        self._history.append((gesture_name, confidence))

        if not self._history:
            return "Unknown", 0.0

        # Tally gesture occurrences
        counts = Counter(g for g, _ in self._history)
        most_common_gesture, count = counts.most_common(1)[0]

        # Check if consensus threshold is met
        required_votes = int(len(self._history) * self.consensus_ratio)
        if count >= max(1, required_votes):
            confs = [c for g, c in self._history if g == most_common_gesture]
            avg_conf = sum(confs) / len(confs)
            return most_common_gesture, round(avg_conf, 2)

        # Fallback to current if no strong consensus
        return gesture_name, confidence

    def clear(self) -> None:
        """Clears the history window."""
        self._history.clear()
