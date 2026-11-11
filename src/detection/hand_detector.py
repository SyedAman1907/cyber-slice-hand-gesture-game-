"""
Module 2: Hand Detection Base & Utilities.
Defines unified data models, landmark indices, smoothing, and rendering logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# Standard 21 MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Skeletal connections (pairs of landmark indices)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb ray
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # Index finger ray
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # Middle finger ray
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # Ring finger ray
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # Pinky finger ray
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    # Transverse palmar arch (connecting knuckle MCP joints)
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
    # Central palmar anchor
    (WRIST, MIDDLE_MCP)
]


@dataclass
class HandLandmark:
    """Normalized landmark coordinate (0.0 to 1.0)."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class DetectedHand:
    """Represents a single detected hand in a frame."""
    landmarks: List[HandLandmark] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    handedness: str = "Right"
    confidence: float = 0.0
    hand_id: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates normalized center of palm using wrist and MCP knuckles."""
        if not self.landmarks or len(self.landmarks) < 21:
            return (0.5, 0.5)
        # Average of wrist and key MCP joints gives stable palm center
        pts = [self.landmarks[WRIST], self.landmarks[INDEX_MCP],
               self.landmarks[MIDDLE_MCP], self.landmarks[PINKY_MCP]]
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)
        return (cx, cy)

    @property
    def index_tip(self) -> Tuple[float, float]:
        """Returns normalized (x, y) of index fingertip."""
        if len(self.landmarks) > INDEX_TIP:
            return (self.landmarks[INDEX_TIP].x, self.landmarks[INDEX_TIP].y)
        return self.center


class BaseHandDetector(ABC):
    """Abstract base class for hand detection implementations."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedHand]:
        """Processes frame and returns detected hands with landmarks."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases underlying model resources."""
        pass

    def set_mirrored(self, mirrored: bool) -> None:
        """Updates mirrored coordinate and handedness interpretation."""
        pass


class LandmarkSmoother:
    """Exponential Moving Average (EMA) smoother to reduce landmark jitter."""

    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self._prev_landmarks: Dict[int, List[HandLandmark]] = {}

    def smooth(self, hand_id: int, landmarks: List[HandLandmark]) -> List[HandLandmark]:
        """Smooths landmark positions using history."""
        if hand_id not in self._prev_landmarks or len(self._prev_landmarks[hand_id]) != len(landmarks):
            self._prev_landmarks[hand_id] = landmarks
            return landmarks

        smoothed: List[HandLandmark] = []
        prev = self._prev_landmarks[hand_id]
        for curr_lm, prev_lm in zip(landmarks, prev):
            sx = self.alpha * curr_lm.x + (1.0 - self.alpha) * prev_lm.x
            sy = self.alpha * curr_lm.y + (1.0 - self.alpha) * prev_lm.y
            sz = self.alpha * curr_lm.z + (1.0 - self.alpha) * prev_lm.z
            smoothed.append(HandLandmark(x=sx, y=sy, z=sz))

        self._prev_landmarks[hand_id] = smoothed
        return smoothed

    def reset(self, hand_id: Optional[int] = None) -> None:
        """Clears smoothing history."""
        if hand_id is None:
            self._prev_landmarks.clear()
        elif hand_id in self._prev_landmarks:
            del self._prev_landmarks[hand_id]


def render_hand_annotations(
    frame: np.ndarray,
    hands: List[DetectedHand],
    draw_bbox: bool = True,
    draw_landmarks: bool = True
) -> np.ndarray:
    """
    Renders visual overlays for detected hands on the video frame.
    Separated from detection logic for clean decoupling.
    """
    out_frame = frame.copy()
    h, w = frame.shape[:2]

    # Palette
    color_bone = (255, 200, 0)      # Cyan / Sky blue in BGR
    color_joint = (0, 255, 255)     # Bright yellow/gold
    color_tip = (50, 50, 255)       # Coral / Red-orange for tips
    color_box = (0, 230, 115)       # Neon green

    tips = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}

    for hand in hands:
        # 1. Draw Bounding Box and Handedness Badge
        if draw_bbox and hand.bbox[2] > 0 and hand.bbox[3] > 0:
            bx, by, bw, bh = hand.bbox
            # Draw sleek corner brackets
            corner_len = min(20, bw // 4, bh // 4)
            t = 2
            # Top-left
            cv2.line(out_frame, (bx, by), (bx + corner_len, by), color_box, t)
            cv2.line(out_frame, (bx, by), (bx, by + corner_len), color_box, t)
            # Top-right
            cv2.line(out_frame, (bx + bw, by), (bx + bw - corner_len, by), color_box, t)
            cv2.line(out_frame, (bx + bw, by), (bx + bw, by + corner_len), color_box, t)
            # Bottom-left
            cv2.line(out_frame, (bx, by + bh), (bx + corner_len, by + bh), color_box, t)
            cv2.line(out_frame, (bx, by + bh), (bx, by + bh - corner_len), color_box, t)
            # Bottom-right
            cv2.line(out_frame, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color_box, t)
            cv2.line(out_frame, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color_box, t)

            # Badge pill
            badge_text = f"{hand.handedness} ({int(hand.confidence * 100)}%)"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pill_y1 = max(0, by - 24)
            cv2.rectangle(out_frame, (bx, pill_y1), (bx + tw + 10, pill_y1 + th + 8), (20, 20, 20), -1)
            cv2.rectangle(out_frame, (bx, pill_y1), (bx + tw + 10, pill_y1 + th + 8), color_box, 1)
            cv2.putText(out_frame, badge_text, (bx + 5, pill_y1 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Skeletal Connections & Landmarks
        if draw_landmarks and len(hand.landmarks) == 21:
            px_pts: List[Tuple[int, int]] = []
            for lm in hand.landmarks:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                px_pts.append((px, py))

            # Bones
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                cv2.line(out_frame, px_pts[p1_idx], px_pts[p2_idx], color_bone, 2, cv2.LINE_AA)

            # Joints
            for idx, pt in enumerate(px_pts):
                if idx in tips:
                    cv2.circle(out_frame, pt, 6, color_tip, -1, cv2.LINE_AA)
                    cv2.circle(out_frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(out_frame, pt, 4, color_joint, -1, cv2.LINE_AA)

    return out_frame
