"""
Settings Screen Module.
Configures camera selection, detection backend, gesture sensitivity,
audio toggles, and game difficulty with real-time persistence.
"""

from typing import Callable, Dict, Tuple
import pygame

from src.ui.components import UIButton, UIPanel
from src.utils.config import ConfigManager


class SettingsScreen:
    """Provides interactive controls to adjust game and vision preferences."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        config_manager: ConfigManager,
        on_back: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts
        self.config = config_manager
        self.on_back = on_back

        # Settings state in memory
        self.camera_idx = self.config.get("camera", "device_index", 0)
        self.mirrored = self.config.get("camera", "mirrored", True)
        self.backend = self.config.get("detection", "backend", "mediapipe")
        self.sensitivity = self.config.get("gestures", "sensitivity", 1.0)
        self.difficulty = self.config.get("game", "difficulty", "Medium")
        self.sound_enabled = self.config.get("game", "sound_enabled", True)

        # Build Interactive Option Buttons
        self._build_buttons()

    def _build_buttons(self) -> None:
        """Constructs option selector buttons."""
        cx = self.width // 2
        btn_w, btn_h = 220, 42
        start_y = 120
        row_gap = 62

        # 1. Camera Index Toggle
        self.btn_camera = UIButton(
            (cx + 40, start_y, btn_w, btn_h),
            f"Device Index: {self.camera_idx}",
            self.fonts["medium"],
            callback=self._cycle_camera
        )

        # 2. Mirror Toggle
        self.btn_mirror = UIButton(
            (cx + 40, start_y + row_gap, btn_w, btn_h),
            "ON" if self.mirrored else "OFF",
            self.fonts["medium"],
            callback=self._toggle_mirror
        )

        # 3. Detection Backend Toggle
        self.btn_backend = UIButton(
            (cx + 40, start_y + row_gap * 2, btn_w, btn_h),
            self.backend.upper(),
            self.fonts["medium"],
            callback=self._toggle_backend
        )

        # 4. Gesture Sensitivity
        self.btn_sens = UIButton(
            (cx + 40, start_y + row_gap * 3, btn_w, btn_h),
            f"{self.sensitivity:.1f}x",
            self.fonts["medium"],
            callback=self._cycle_sensitivity
        )

        # 5. Difficulty Level
        self.btn_diff = UIButton(
            (cx + 40, start_y + row_gap * 4, btn_w, btn_h),
            self.difficulty,
            self.fonts["medium"],
            callback=self._cycle_difficulty
        )

        # 6. Sound Toggle
        self.btn_sound = UIButton(
            (cx + 40, start_y + row_gap * 5, btn_w, btn_h),
            "ENABLED" if self.sound_enabled else "MUTED",
            self.fonts["medium"],
            callback=self._toggle_sound
        )

        # 7. Save & Return
        self.btn_save_back = UIButton(
            (cx - 150, self.height - 80, 300, 50),
            "SAVE & RETURN",
            self.fonts["medium"],
            bg_color=(20, 80, 120),
            hover_color=(0, 210, 255),
            border_color=(0, 180, 216),
            callback=self._save_and_exit
        )

        self.buttons = [
            self.btn_camera,
            self.btn_mirror,
            self.btn_backend,
            self.btn_sens,
            self.btn_diff,
            self.btn_sound,
            self.btn_save_back
        ]

    def _cycle_camera(self) -> None:
        self.camera_idx = (self.camera_idx + 1) % 3
        self.btn_camera.text = f"Device Index: {self.camera_idx}"

    def _toggle_mirror(self) -> None:
        self.mirrored = not self.mirrored
        self.btn_mirror.text = "ON" if self.mirrored else "OFF"

    def _toggle_backend(self) -> None:
        self.backend = "yolo" if self.backend == "mediapipe" else "mediapipe"
        self.btn_backend.text = self.backend.upper()

    def _cycle_sensitivity(self) -> None:
        presets = [0.8, 1.0, 1.3, 1.6]
        idx = (presets.index(self.sensitivity) + 1) % len(presets) if self.sensitivity in presets else 1
        self.sensitivity = presets[idx]
        self.btn_sens.text = f"{self.sensitivity:.1f}x"

    def _cycle_difficulty(self) -> None:
        diffs = ["Easy", "Medium", "Hard"]
        idx = (diffs.index(self.difficulty) + 1) % len(diffs)
        self.difficulty = diffs[idx]
        self.btn_diff.text = self.difficulty

    def _toggle_sound(self) -> None:
        self.sound_enabled = not self.sound_enabled
        self.btn_sound.text = "ENABLED" if self.sound_enabled else "MUTED"

    def _save_and_exit(self) -> None:
        """Persists choices to config and returns."""
        self.config.set("camera", "device_index", self.camera_idx)
        self.config.set("camera", "mirrored", self.mirrored)
        self.config.set("detection", "backend", self.backend)
        self.config.set("gestures", "sensitivity", self.sensitivity)
        self.config.set("game", "difficulty", self.difficulty)
        self.config.set("game", "sound_enabled", self.sound_enabled)
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
                self.btn_save_back.trigger()
                return
            elif event.key == pygame.K_ESCAPE:
                self.on_back()
                return
        for btn in self.buttons:
            btn.handle_event(event)

    def draw(self, surface: pygame.Surface) -> None:
        """Renders settings menu."""
        surface.fill((12, 16, 26))

        cx = self.width // 2

        # Title
        t_surf = self.fonts["large"].render("SYSTEM & GAME SETTINGS", True, (0, 235, 255))
        surface.blit(t_surf, t_surf.get_rect(center=(cx, 55)))

        # Card container
        panel_w, panel_h = 700, 420
        panel_rect = (cx - panel_w // 2, 95, panel_w, panel_h)
        UIPanel.draw(surface, panel_rect, bg_color=(16, 22, 38, 230), border_color=(45, 65, 95))

        # Labels on Left, Buttons on Right
        start_y = 135
        row_gap = 62
        labels = [
            ("Camera Device Index", "Selects physical or virtual webcam"),
            ("Mirrored Camera Feed", "Horizontally flips video for natural motion"),
            ("Detection Backend", "MediaPipe (Landmarks) or Ultralytics YOLO"),
            ("Gesture Sensitivity", "Scales extension & motion thresholds"),
            ("Game Difficulty", "Adjusts fruit speed, waves & bomb frequency"),
            ("Sound Effects", "Procedurally synthesized audio feedback")
        ]

        for i, (main_lbl, sub_lbl) in enumerate(labels):
            ly = start_y + i * row_gap
            m_surf = self.fonts["medium"].render(main_lbl, True, (240, 240, 250))
            surface.blit(m_surf, (cx - panel_w // 2 + 40, ly - 8))

            s_surf = self.fonts["small"].render(sub_lbl, True, (130, 155, 190))
            surface.blit(s_surf, (cx - panel_w // 2 + 40, ly + 16))

        # Buttons
        for btn in self.buttons:
            btn.draw(surface)
