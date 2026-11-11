"""
Live Performance Diagnostics HUD Panel Module.
Displays real-time hardware telemetry:
- Camera FPS & Processing FPS
- Model inference time (ms)
- Gesture recognition time (ms)
- Measured input-to-display latency (ms)
- Tracking status (Tracking / Searching / Lost) & Hand confidence
- Active gesture & Normalized cursor coordinates
"""

from typing import Dict, Optional, Tuple
import pygame

from src.detection.hand_tracker import TrackingStatus
from src.gestures.gesture_recognizer import RecognizedGesture
from src.ui.components import UIBadge, UIPanel


class DiagnosticsPanel:
    """
    Renders non-intrusive, real-time computer vision performance telemetry.
    Can be toggled with the 'D' key or displayed in debug/calibration screens.
    """

    def __init__(self, screen_w: int, screen_h: int, font: pygame.font.Font):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.font = font
        self.is_visible = True

    @property
    def enabled(self) -> bool:
        """Returns whether diagnostics HUD is enabled."""
        return self.is_visible

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self.is_visible = bool(val)

    def toggle(self) -> None:
        """Toggles diagnostics visibility."""
        self.is_visible = not self.is_visible

    def draw(
        self,
        surface: pygame.Surface,
        camera_fps: float,
        processing_fps: float,
        inference_time_ms: float,
        gesture_time_ms: float,
        latency_ms: float,
        tracking_status: TrackingStatus,
        hand_confidence: float,
        gesture: RecognizedGesture,
        cursor_pos: Tuple[float, float],
        pos: Tuple[int, int] = (15, 110)
    ) -> None:
        """Renders the telemetry panel."""
        if not self.is_visible:
            return

        panel_w = 265
        panel_h = 210
        x, y = pos

        # Semi-transparent dark glass panel
        UIPanel.draw(
            surface,
            (x, y, panel_w, panel_h),
            bg_color=(10, 14, 24, 220),
            border_color=(0, 200, 240, 180),
            border_radius=8
        )

        # Header
        hdr_txt = "SYSTEM DIAGNOSTICS [D]"
        hdr_surf = self.font.render(hdr_txt, True, (0, 235, 255))
        surface.blit(hdr_surf, (x + 12, y + 10))
        pygame.draw.line(surface, (35, 50, 75), (x + 10, y + 30), (x + panel_w - 10, y + 30), 1)

        # Status badge dot color
        if tracking_status == TrackingStatus.TRACKING:
            status_color = (0, 255, 120)  # Green
        elif tracking_status == TrackingStatus.SEARCHING:
            status_color = (255, 200, 0)  # Orange
        else:
            status_color = (255, 70, 70)   # Red

        # Format metrics
        metrics = [
            ("Tracking Status", tracking_status.value, status_color),
            ("Camera Hardware", f"{camera_fps:.1f} FPS", (220, 225, 240)),
            ("Pipeline Rate", f"{processing_fps:.1f} FPS", (0, 235, 255)),
            ("Model Inference", f"{inference_time_ms:.1f} ms", (255, 215, 0)),
            ("Gesture Engine", f"{gesture_time_ms:.2f} ms", (200, 220, 245)),
            ("Input Latency", f"{latency_ms:.1f} ms", (0, 255, 150) if latency_ms < 80 else (255, 140, 0)),
            ("Hand Confidence", f"{int(hand_confidence * 100)}%", (210, 220, 240)),
            ("Active Gesture", f"{gesture.name[:16]}", (255, 215, 0)),
            ("Cursor Norm", f"({cursor_pos[0]:.2f}, {cursor_pos[1]:.2f})", (140, 175, 210))
        ]

        row_y = y + 38
        line_h = 18
        for label, val, color in metrics:
            lbl_surf = self.font.render(label + ":", True, (130, 155, 185))
            surface.blit(lbl_surf, (x + 12, row_y))

            val_surf = self.font.render(val, True, color)
            val_rect = val_surf.get_rect(topright=(x + panel_w - 12, row_y))
            surface.blit(val_surf, val_rect)

            row_y += line_h
