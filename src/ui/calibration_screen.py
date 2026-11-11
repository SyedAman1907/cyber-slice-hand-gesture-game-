"""
Hand Calibration & Testing Screen Module.
Interactive tuning environment providing:
- Hand bounding guidance region
- 21-landmark inspection
- Lighting quality and camera distance assessment
- Live sensitivity, One Euro smoothing, and control-point adjustments
- Left/Right coordinate mapping verification
"""

import math
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pygame

from src.detection.hand_detector import DetectedHand, render_hand_annotations
from src.detection.hand_tracker import HandTracker, TrackingStatus
from src.gestures.gesture_recognizer import RecognizedGesture
from src.ui.components import UIBadge, UIButton, UIPanel, cvframe_to_pygame_surface
from src.utils.config import ConfigManager


class CalibrationScreen:
    """Provides real-time interactive camera calibration and latency tuning."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        config_manager: ConfigManager,
        hand_tracker: HandTracker,
        on_back: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts
        self.config = config_manager
        self.tracker = hand_tracker
        self.on_back = on_back

        # Live Configurable Parameters
        self.sensitivity = self.config.get("gestures", "sensitivity", 1.0)
        self.control_point = self.config.get("gestures", "control_point", "index_tip")
        self.slicing_mode = self.config.get("game", "slicing_mode", "motion")
        self.min_cutoff = self.config.get("gestures", "one_euro_min_cutoff", 1.2)
        self.beta = self.config.get("gestures", "one_euro_beta", 15.0)
        self.timeout = self.config.get("gestures", "tracking_loss_timeout", 0.14)

        # Gestures tested during calibration session
        self.calibrated_gestures = set()

        # UI Buttons
        self._build_controls()

    def _build_controls(self) -> None:
        cx = self.width // 2
        right_x = 640
        btn_y = 110
        btn_w, btn_h = 240, 38
        gap = 52

        # 1. Control Point Toggle
        self.btn_control_pt = UIButton(
            (right_x + 360, btn_y, btn_w, btn_h),
            "Index Tip" if self.control_point == "index_tip" else "Palm Center",
            self.fonts["small"],
            callback=self._toggle_control_pt
        )

        # 2. Slicing Mode Toggle
        self.btn_slice_mode = UIButton(
            (right_x + 360, btn_y + gap, btn_w, btn_h),
            self.slicing_mode.upper(),
            self.fonts["small"],
            callback=self._toggle_slice_mode
        )

        # 3. Sensitivity Increments
        self.btn_sens_down = UIButton((right_x + 360, btn_y + gap * 2, 60, btn_h), "-", self.fonts["medium"], callback=self._sens_down)
        self.btn_sens_up = UIButton((right_x + 540, btn_y + gap * 2, 60, btn_h), "+", self.fonts["medium"], callback=self._sens_up)

        # 4. Smoothing Filter Cutoff
        self.btn_smooth_down = UIButton((right_x + 360, btn_y + gap * 3, 60, btn_h), "-", self.fonts["medium"], callback=self._smooth_down)
        self.btn_smooth_up = UIButton((right_x + 540, btn_y + gap * 3, 60, btn_h), "+", self.fonts["medium"], callback=self._smooth_up)

        # 5. Tracking Loss Timeout
        self.btn_timeout_down = UIButton((right_x + 360, btn_y + gap * 4, 60, btn_h), "-", self.fonts["medium"], callback=self._timeout_down)
        self.btn_timeout_up = UIButton((right_x + 540, btn_y + gap * 4, 60, btn_h), "+", self.fonts["medium"], callback=self._timeout_up)

        # 6. Apply & Return Button
        self.btn_save_exit = UIButton(
            (self.width // 2 - 150, self.height - 70, 300, 48),
            "APPLY & SAVE SETTINGS",
            self.fonts["medium"],
            bg_color=(20, 80, 120),
            hover_color=(0, 210, 255),
            border_color=(0, 180, 216),
            callback=self._save_and_exit
        )

        self.buttons = [
            self.btn_control_pt,
            self.btn_slice_mode,
            self.btn_sens_down, self.btn_sens_up,
            self.btn_smooth_down, self.btn_smooth_up,
            self.btn_timeout_down, self.btn_timeout_up,
            self.btn_save_exit
        ]

    def _toggle_control_pt(self) -> None:
        self.control_point = "palm_center" if self.control_point == "index_tip" else "index_tip"
        self.btn_control_pt.text = "Index Tip" if self.control_point == "index_tip" else "Palm Center"
        self.tracker.update_settings(control_point=self.control_point)

    def _toggle_slice_mode(self) -> None:
        modes = ["motion", "pinch", "fist", "point", "always"]
        idx = (modes.index(self.slicing_mode) + 1) % len(modes) if self.slicing_mode in modes else 0
        self.slicing_mode = modes[idx]
        self.btn_slice_mode.text = self.slicing_mode.upper()

    def _sens_down(self) -> None:
        self.sensitivity = max(0.5, round(self.sensitivity - 0.1, 1))
        self.tracker.update_settings(sensitivity=self.sensitivity)

    def _sens_up(self) -> None:
        self.sensitivity = min(2.5, round(self.sensitivity + 0.1, 1))
        self.tracker.update_settings(sensitivity=self.sensitivity)

    def _smooth_down(self) -> None:
        self.min_cutoff = max(0.4, round(self.min_cutoff - 0.2, 1))
        self.tracker.update_settings(min_cutoff=self.min_cutoff, beta=self.beta)

    def _smooth_up(self) -> None:
        self.min_cutoff = min(3.0, round(self.min_cutoff + 0.2, 1))
        self.tracker.update_settings(min_cutoff=self.min_cutoff, beta=self.beta)

    def _timeout_down(self) -> None:
        self.timeout = max(0.06, round(self.timeout - 0.02, 2))
        self.tracker.update_settings(timeout=self.timeout)

    def _timeout_up(self) -> None:
        self.timeout = min(0.30, round(self.timeout + 0.02, 2))
        self.tracker.update_settings(timeout=self.timeout)

    def _save_and_exit(self) -> None:
        self.config.set("gestures", "sensitivity", self.sensitivity)
        self.config.set("gestures", "control_point", self.control_point)
        self.config.set("game", "slicing_mode", self.slicing_mode)
        self.config.set("gestures", "one_euro_min_cutoff", self.min_cutoff)
        self.config.set("gestures", "one_euro_beta", self.beta)
        self.config.set("gestures", "tracking_loss_timeout", self.timeout)
        self.config.save()
        self.on_back()

    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        for btn in self.buttons:
            btn.update(mouse_pos, dt)

    def trigger_hovered_button(self, mouse_pos: Tuple[int, int]) -> bool:
        """Triggers hovered button via pinch gesture."""
        for btn in self.buttons:
            if btn.rect.collidepoint(mouse_pos) or btn.is_hovered:
                return btn.trigger()
        return False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.btn_save_exit.trigger()
                return
            elif event.key == pygame.K_ESCAPE:
                self.on_back()
                return
        for btn in self.buttons:
            btn.handle_event(event)

    def draw(
        self,
        surface: pygame.Surface,
        raw_frame: Optional[np.ndarray],
        detected_hands: List[DetectedHand],
        gesture: RecognizedGesture,
        tracking_status: TrackingStatus,
        camera_fps: float = 30.0,
        latency_ms: float = 40.0
    ) -> None:
        """Renders the comprehensive calibration UI."""
        surface.fill((12, 16, 26))

        # Title
        t_surf = self.fonts["large"].render("CAMERA & HAND TRACKING CALIBRATION", True, (0, 235, 255))
        surface.blit(t_surf, (40, 20))

        sub_txt = "Verify lighting quality, zero-lag 21-landmark tracking, and fine-tune responsive One Euro parameters."
        s_surf = self.fonts["small"].render(sub_txt, True, (140, 170, 205))
        surface.blit(s_surf, (40, 56))

        # 1. Left: Large Camera Viewport with Guided Placement Region
        cam_w, cam_h = 560, 420
        cam_x, cam_y = 40, 95

        lighting_eval = "Lighting Unknown"
        lighting_color = (180, 180, 180)
        distance_eval = "Searching for Hand..."
        distance_color = (200, 200, 200)

        if raw_frame is not None:
            # Measure mean lighting brightness
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))

            if mean_brightness < 45:
                lighting_eval = f"Poor Lighting: Too Dark ({int(mean_brightness)}/255)"
                lighting_color = (255, 75, 75)
            elif mean_brightness > 220:
                lighting_eval = f"Glare Warning: Overexposed ({int(mean_brightness)}/255)"
                lighting_color = (255, 140, 0)
            else:
                lighting_eval = f"Lighting Optimal ({int(mean_brightness)}/255)"
                lighting_color = (0, 255, 120)

            # Render landmarks
            annotated = render_hand_annotations(raw_frame, detected_hands, draw_bbox=True, draw_landmarks=True)

            # Draw optimal Hand Placement Guide Box in center of camera frame
            cw, ch = raw_frame.shape[1], raw_frame.shape[0]
            gx1, gy1 = int(cw * 0.25), int(ch * 0.15)
            gx2, gy2 = int(cw * 0.75), int(ch * 0.85)
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 235, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, "Optimal Hand Zone", (gx1 + 10, gy1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 235, 255), 1, cv2.LINE_AA)

            # Evaluate distance / scale and frame boundary containment
            if detected_hands:
                h = detected_hands[0]
                if len(h.landmarks) >= 21:
                    xs = [lm.x for lm in h.landmarks]
                    ys = [lm.y for lm in h.landmarks]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    if min_x < 0.05 or max_x > 0.95 or min_y < 0.05 or max_y > 0.95:
                        distance_eval = "Hand Near Edge: Keep Centered in Zone"
                        distance_color = (255, 140, 0)
                    else:
                        scale = math.hypot(h.landmarks[0].x - h.landmarks[9].x, h.landmarks[0].y - h.landmarks[9].y)
                        if scale < 0.12:
                            distance_eval = "Hand Too Far: Move Closer"
                            distance_color = (255, 140, 0)
                        elif scale > 0.45:
                            distance_eval = "Hand Too Close: Move Back"
                            distance_color = (255, 140, 0)
                        else:
                            distance_eval = f"Distance Optimal (Scale {scale:.2f})"
                            distance_color = (0, 255, 120)

            cam_surf = cvframe_to_pygame_surface(annotated, (cam_w, cam_h))
            surface.blit(cam_surf, (cam_x, cam_y))
            pygame.draw.rect(surface, (0, 200, 240), (cam_x - 2, cam_y - 2, cam_w + 4, cam_h + 4), width=2, border_radius=8)

            # Draw target cursor on camera preview for direct visual verification
            cur_cam_x = cam_x + int(np.clip(gesture.cursor_pos[0], 0.0, 1.0) * cam_w)
            cur_cam_y = cam_y + int(np.clip(gesture.cursor_pos[1], 0.0, 1.0) * cam_h)
            pygame.draw.circle(surface, (0, 255, 255), (cur_cam_x, cur_cam_y), 9, 2)
            pygame.draw.circle(surface, (255, 255, 255), (cur_cam_x, cur_cam_y), 4)

            # Camera badges
            UIBadge.draw(surface, (cam_x + 12, cam_y + 12), f"Hardware: {int(camera_fps)} FPS", self.fonts["small"])
            UIBadge.draw(surface, (cam_x + 170, cam_y + 12), f"Latency: {int(latency_ms)} ms", self.fonts["small"], dot_color=(0, 255, 120))
        else:
            UIPanel.draw(surface, (cam_x, cam_y, cam_w, cam_h), bg_color=(15, 20, 32), border_color=(200, 60, 60))

        # Bottom Environmental Assessment Card
        env_card = (cam_x, cam_y + cam_h + 14, cam_w, 85)
        UIPanel.draw(surface, env_card, bg_color=(16, 22, 38, 220), border_color=(40, 60, 90))

        surface.blit(self.fonts["small"].render("ENVIRONMENT QUALITY CHECK:", True, (0, 235, 255)), (cam_x + 16, cam_y + cam_h + 24))
        surface.blit(self.fonts["medium"].render(lighting_eval, True, lighting_color), (cam_x + 16, cam_y + cam_h + 42))
        surface.blit(self.fonts["small"].render(distance_eval, True, distance_color), (cam_x + 16, cam_y + cam_h + 68))

        # 2. Right: Interactive Tuning Controls & Telemetry
        right_x = cam_x + cam_w + 30
        panel_w = self.width - right_x - 40
        panel_h = 520

        UIPanel.draw(surface, (right_x, cam_y, panel_w, panel_h), bg_color=(15, 20, 34, 220), border_color=(45, 65, 95))

        # Section Header
        surface.blit(self.fonts["medium"].render("TRACKING & FILTER PARAMETERS", True, (245, 245, 255)), (right_x + 20, cam_y + 16))

        labels = [
            ("Target Control Point", "Landmark followed by slicing cursor"),
            ("Slicing Activation Mode", "Motion | Pinch | Fist | Point | Always"),
            ("Cursor Sensitivity", f"Current multiplier: {self.sensitivity:.1f}x"),
            ("Smoothing Filter Cutoff", f"Current min_cutoff: {self.min_cutoff:.1f} Hz"),
            ("Tracking-Loss Timeout", f"Current grace timeout: {int(self.timeout * 1000)} ms")
        ]

        row_y = cam_y + 54
        gap = 52
        for i, (main_lbl, desc_lbl) in enumerate(labels):
            surface.blit(self.fonts["medium"].render(main_lbl, True, (220, 230, 245)), (right_x + 20, row_y + i * gap))
            surface.blit(self.fonts["small"].render(desc_lbl, True, (130, 155, 190)), (right_x + 20, row_y + i * gap + 22))

        # Middle Value displays for (+ / -) buttons
        surface.blit(self.fonts["medium"].render(f"{self.sensitivity:.1f}x", True, (0, 235, 255)), (right_x + 445, cam_y + 54 + gap * 2 + 6))
        surface.blit(self.fonts["medium"].render(f"{self.min_cutoff:.1f} Hz", True, (0, 235, 255)), (right_x + 435, cam_y + 54 + gap * 3 + 6))
        surface.blit(self.fonts["medium"].render(f"{int(self.timeout * 1000)} ms", True, (0, 235, 255)), (right_x + 435, cam_y + 54 + gap * 4 + 6))

        # Record calibrated gesture
        if gesture.name in ("Open Palm", "Closed Fist", "Thumbs Up", "Index Finger Pointing", "Pinch Gesture", "Swipe Left", "Swipe Right"):
            self.calibrated_gestures.add(gesture.name)

        # Live Anatomical Telemetry Card
        debug_box = (right_x + 15, cam_y + 320, panel_w - 30, 180)
        UIPanel.draw(surface, debug_box, bg_color=(12, 16, 26, 230), border_color=(35, 50, 75))

        surface.blit(self.fonts["small"].render("LIVE ANATOMICAL TELEMETRY & VERIFICATION", True, (0, 235, 255)), (right_x + 28, cam_y + 328))

        # Show finger states
        dbg = gesture.debug_info
        f_states = dbg.get("finger_states", {})
        thumb_str = "EXT" if f_states.get("thumb") else "CURL"
        idx_str = "EXT" if f_states.get("index") else "CURL"
        mid_str = "EXT" if f_states.get("middle") else "CURL"
        rng_str = "EXT" if f_states.get("ring") else "CURL"
        pnk_str = "EXT" if f_states.get("pinky") else "CURL"

        f_line = f"Fingers: T:{thumb_str} | I:{idx_str} | M:{mid_str} | R:{rng_str} | P:{pnk_str}"
        surface.blit(self.fonts["small"].render(f_line, True, (240, 240, 245)), (right_x + 28, cam_y + 350))

        g_line = f"Active Gesture: {gesture.name} ({int(gesture.confidence * 100)}%)"
        surface.blit(self.fonts["medium"].render(g_line, True, (255, 215, 0)), (right_x + 28, cam_y + 372))

        hand_info = f"Hand: {gesture.handedness} | Status: {tracking_status.value} | Scale: {dbg.get('hand_scale', 0.0)}"
        surface.blit(self.fonts["small"].render(hand_info, True, (150, 180, 210)), (right_x + 28, cam_y + 400))

        rec_info = f"Latency: {dbg.get('rec_time_ms', 0.0):.2f} ms | Cursor: ({gesture.cursor_pos[0]:.2f}, {gesture.cursor_pos[1]:.2f})"
        surface.blit(self.fonts["small"].render(rec_info, True, (130, 160, 190)), (right_x + 28, cam_y + 422))

        # Gesture Verification Checklist Pills
        test_keys = [
            ("Pinch", "Pinch Gesture" in self.calibrated_gestures),
            ("Palm", "Open Palm" in self.calibrated_gestures),
            ("Fist", "Closed Fist" in self.calibrated_gestures),
            ("Point", "Index Finger Pointing" in self.calibrated_gestures),
            ("Swipe", ("Swipe Left" in self.calibrated_gestures or "Swipe Right" in self.calibrated_gestures))
        ]
        pill_x = right_x + 28
        pill_y = cam_y + 452
        for g_label, is_ok in test_keys:
            color = (0, 255, 120) if is_ok else (120, 135, 155)
            badge_txt = f"{'[OK]' if is_ok else '[ ]'} {g_label}"
            UIBadge.draw(surface, (pill_x, pill_y), badge_txt, self.fonts["small"], dot_color=color)
            pill_x += 92

        # 3. Draw UI Buttons
        for btn in self.buttons:
            btn.draw(surface)
