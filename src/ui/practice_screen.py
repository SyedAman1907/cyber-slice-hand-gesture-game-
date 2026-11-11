"""
Practice Mode Screen Module.
Dedicated calibration and practice environment for users to test and master
all 8 gestures in real-time with visual landmark overlays and validation checklist.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import pygame

from src.detection.hand_detector import DetectedHand, render_hand_annotations
from src.gestures.gesture_recognizer import GESTURE_DESCRIPTIONS, RecognizedGesture
from src.ui.components import UIBadge, UIButton, UIPanel, cvframe_to_pygame_surface

ALL_PRACTICE_GESTURES = [
    ("Open Palm", "Spread all 5 fingers wide", "Start Game / Sweep"),
    ("Closed Fist", "Clench all fingers tightly into palm", "Stop / Shield"),
    ("Thumbs Up", "Extend thumb upright, other fingers curled", "Pause / Resume"),
    ("Index Finger Pointing", "Extend only index finger forward", "Cursor & Slice"),
    ("Two-Finger Gesture", "Extend index and middle fingers together", "Dual Blade"),
    ("Pinch Gesture", "Touch thumb tip and index fingertip together", "Grab / Click"),
    ("Swipe Left", "Quickly sweep hand from right to left", "Left Sweep"),
    ("Swipe Right", "Quickly sweep hand from left to right", "Right Sweep")
]


class PracticeScreen:
    """Provides real-time interactive feedback for all 8 gestures."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        on_back: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts
        self.on_back = on_back

        # Tested gestures tally
        self.tested_gestures: Set[str] = set()
        self.gesture_counts: Dict[str, int] = {g[0]: 0 for g in ALL_PRACTICE_GESTURES}

        # Back Button
        self.back_button = UIButton(
            (40, screen_h - 65, 200, 46),
            "BACK TO MENU",
            fonts["medium"],
            callback=on_back
        )
        self.reset_button = UIButton(
            (260, screen_h - 65, 180, 46),
            "RESET STATS",
            fonts["medium"],
            callback=self.reset_checklist
        )

    def reset_checklist(self) -> None:
        """Clears tested gestures checklist."""
        self.tested_gestures.clear()
        for k in self.gesture_counts:
            self.gesture_counts[k] = 0

    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Updates UI buttons."""
        self.back_button.update(mouse_pos, dt)
        self.reset_button.update(mouse_pos, dt)

    def trigger_hovered_button(self, mouse_pos: Tuple[int, int]) -> bool:
        """Triggers hovered button via pinch gesture."""
        for btn in (self.back_button, self.reset_button):
            if btn.rect.collidepoint(mouse_pos) or btn.is_hovered:
                return btn.trigger()
        return False

    def handle_event(self, event: pygame.event.Event) -> None:
        """Dispatches button clicks and keyboard shortcuts."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.back_button.trigger()
                return
        self.back_button.handle_event(event)
        self.reset_button.handle_event(event)

    def record_gesture(self, gesture_name: str) -> None:
        """Records detection of a gesture in practice mode."""
        if gesture_name in self.gesture_counts:
            self.tested_gestures.add(gesture_name)
            self.gesture_counts[gesture_name] += 1

    def draw(
        self,
        surface: pygame.Surface,
        raw_frame: Optional[np.ndarray],
        detected_hands: List[DetectedHand],
        current_gesture: RecognizedGesture,
        camera_fps: float = 30.0
    ) -> None:
        """Renders the practice mode dashboard."""
        surface.fill((12, 16, 26))

        # 1. Header
        header_y = 20
        title_surf = self.fonts["large"].render("GESTURE CALIBRATION & PRACTICE", True, (0, 235, 255))
        surface.blit(title_surf, (40, header_y))

        sub_text = "Perform each hand gesture to verify recognition and landmark tracking accuracy."
        sub_surf = self.fonts["small"].render(sub_text, True, (150, 175, 210))
        surface.blit(sub_surf, (40, header_y + 36))

        # 2. Left Side: Large Camera Viewport
        cam_w, cam_h = 560, 420
        cam_x, cam_y = 40, 85

        if raw_frame is not None:
            annotated = render_hand_annotations(raw_frame, detected_hands, draw_bbox=True, draw_landmarks=True)
            cam_surf = cvframe_to_pygame_surface(annotated, (cam_w, cam_h))
            surface.blit(cam_surf, (cam_x, cam_y))
            pygame.draw.rect(surface, (0, 200, 240), (cam_x - 2, cam_y - 2, cam_w + 4, cam_h + 4), width=2, border_radius=8)

            # Overlay FPS badge
            UIBadge.draw(surface, (cam_x + 12, cam_y + 12), f"LIVE: {int(camera_fps)} FPS", self.fonts["small"])
        else:
            UIPanel.draw(surface, (cam_x, cam_y, cam_w, cam_h), bg_color=(15, 20, 32), border_color=(180, 50, 50))
            no_cam = self.fonts["medium"].render("Camera Not Detected", True, (220, 80, 80))
            surface.blit(no_cam, no_cam.get_rect(center=(cam_x + cam_w // 2, cam_y + cam_h // 2)))

        # Bottom Live Feedback Card below camera
        card_rect = (cam_x, cam_y + cam_h + 16, cam_w, 75)
        UIPanel.draw(surface, card_rect, bg_color=(16, 22, 36, 220), border_color=(40, 60, 90))

        active_label = f"Current Detected: {current_gesture.name}"
        active_color = (0, 255, 150) if current_gesture.name != "Unknown" else (160, 160, 170)
        lbl_surf = self.fonts["medium"].render(active_label, True, active_color)
        surface.blit(lbl_surf, (cam_x + 16, cam_y + cam_h + 26))

        conf_txt = f"Confidence: {int(current_gesture.confidence * 100)}% | Action: {current_gesture.action}"
        conf_surf = self.fonts["small"].render(conf_txt, True, (140, 175, 210))
        surface.blit(conf_surf, (cam_x + 16, cam_y + cam_h + 54))

        # 3. Right Side: 8-Gesture Interactive Checklist
        checklist_x = cam_x + cam_w + 30
        checklist_w = self.width - checklist_x - 40
        checklist_h = 515
        checklist_panel = (checklist_x, cam_y, checklist_w, checklist_h)
        UIPanel.draw(surface, checklist_panel, bg_color=(15, 20, 34, 200), border_color=(40, 55, 80))

        # Checklist Title & Progress
        completed_count = len(self.tested_gestures)
        pct = int((completed_count / len(ALL_PRACTICE_GESTURES)) * 100)
        ch_title = f"GESTURE CHECKLIST ({completed_count}/{len(ALL_PRACTICE_GESTURES)} - {pct}%)"
        ch_surf = self.fonts["medium"].render(ch_title, True, (240, 240, 250))
        surface.blit(ch_surf, (checklist_x + 20, cam_y + 16))

        # Draw each gesture item row
        row_y = cam_y + 54
        row_h = 50
        for name, instruct, game_act in ALL_PRACTICE_GESTURES:
            is_active = (current_gesture.name == name)
            is_checked = (name in self.tested_gestures)

            row_rect = pygame.Rect(checklist_x + 15, row_y, checklist_w - 30, row_h)

            # Row background
            if is_active:
                bg = (20, 60, 85)
                border = (0, 235, 255)
            elif is_checked:
                bg = (18, 35, 40)
                border = (0, 180, 120)
            else:
                bg = (18, 22, 35)
                border = (35, 45, 65)

            pygame.draw.rect(surface, bg, row_rect, border_radius=8)
            pygame.draw.rect(surface, border, row_rect, width=1, border_radius=8)

            # Checkmark Icon or Box
            check_box = pygame.Rect(row_rect.x + 12, row_rect.y + 14, 22, 22)
            if is_checked:
                pygame.draw.rect(surface, (0, 220, 120), check_box, border_radius=4)
                chk_sym = self.fonts["small"].render("✓", True, (0, 0, 0))
                surface.blit(chk_sym, chk_sym.get_rect(center=check_box.center))
            else:
                pygame.draw.rect(surface, (50, 65, 85), check_box, width=2, border_radius=4)

            # Gesture Name & Description
            g_surf = self.fonts["medium"].render(name, True, (255, 255, 255) if is_active else (210, 215, 225))
            surface.blit(g_surf, (row_rect.x + 44, row_rect.y + 8))

            desc_surf = self.fonts["small"].render(f"{instruct}  →  [{game_act}]", True, (130, 160, 195))
            surface.blit(desc_surf, (row_rect.x + 44, row_rect.y + 28))

            # Hit tally badge
            tally_txt = f"x{self.gesture_counts[name]}"
            tally_surf = self.fonts["small"].render(tally_txt, True, (0, 210, 255) if is_checked else (100, 120, 140))
            surface.blit(tally_surf, (row_rect.right - 45, row_rect.y + 16))

            row_y += row_h + 7

        # 4. Buttons at bottom
        self.back_button.draw(surface)
        self.reset_button.draw(surface)
