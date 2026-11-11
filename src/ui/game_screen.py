"""
Game Screen UI Module.
Renders the in-game HUD, Picture-in-Picture (PIP) camera preview with
skeletal landmarks, lives hearts, score cards, and pause overlay modal.
"""

from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pygame

from src.detection.hand_detector import DetectedHand, render_hand_annotations
from src.detection.hand_tracker import TrackingStatus
from src.game.game_engine import GameEngine, GameState
from src.gestures.gesture_recognizer import RecognizedGesture
from src.ui.components import UIBadge, UIButton, UIPanel, cvframe_to_pygame_surface
from src.ui.diagnostics import DiagnosticsPanel


class GameScreenUI:
    """Renders game HUD overlays, live camera PIP, diagnostics, and pause modals."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        on_resume: Callable[[], None],
        on_restart: Callable[[], None],
        on_menu: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts
        self.on_resume = on_resume
        self.on_restart = on_restart
        self.on_menu = on_menu

        # Performance Diagnostics Panel
        self.diagnostics = DiagnosticsPanel(screen_w, screen_h, fonts["small"])

        # PIP Camera Preview Dimensions
        self.pip_w = 220
        self.pip_h = 165
        self.pip_x = screen_w - self.pip_w - 20
        self.pip_y = 20

        # Pause Modal Buttons
        btn_w, btn_h = 240, 48
        modal_cx = screen_w // 2 - btn_w // 2
        modal_cy = screen_h // 2

        self.pause_buttons = [
            UIButton(
                (modal_cx, modal_cy - 40, btn_w, btn_h),
                "RESUME",
                fonts["medium"],
                bg_color=(20, 80, 120),
                hover_color=(0, 210, 255),
                border_color=(0, 180, 216),
                callback=on_resume
            ),
            UIButton(
                (modal_cx, modal_cy + 20, btn_w, btn_h),
                "RESTART",
                fonts["medium"],
                callback=on_restart
            ),
            UIButton(
                (modal_cx, modal_cy + 80, btn_w, btn_h),
                "MAIN MENU",
                fonts["medium"],
                hover_color=(220, 50, 70),
                callback=on_menu
            )
        ]

    def update(self, mouse_pos: Tuple[int, int], dt: float, is_paused: bool) -> None:
        """Updates pause modal buttons if active."""
        if is_paused:
            for btn in self.pause_buttons:
                btn.update(mouse_pos, dt)

    def handle_event(self, event: pygame.event.Event, is_paused: bool) -> None:
        """Handles clicks on pause modal buttons and toggles diagnostics."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            self.diagnostics.toggle()

        if is_paused:
            for btn in self.pause_buttons:
                btn.handle_event(event)

    def trigger_hovered_button(self, mouse_pos: Tuple[int, int]) -> bool:
        """Triggers hovered pause button via pinch gesture."""
        for btn in self.pause_buttons:
            if btn.rect.collidepoint(mouse_pos) or btn.is_hovered:
                return btn.trigger()
        return False

    def draw_hud(
        self,
        surface: pygame.Surface,
        game: GameEngine,
        raw_frame: Optional[np.ndarray],
        detected_hands: List[DetectedHand],
        gesture: RecognizedGesture,
        camera_fps: float = 30.0,
        backend_name: str = "MediaPipe",
        processing_fps: float = 60.0,
        inference_time_ms: float = 12.0,
        gesture_time_ms: float = 0.5,
        latency_ms: float = 35.0,
        tracking_status: TrackingStatus = TrackingStatus.TRACKING,
        hand_confidence: float = 0.90
    ) -> None:
        """Renders the entire game screen HUD and live diagnostics."""
        # 1. Top-Left: Score and High Score Panel
        score_panel = (20, 20, 230, 80)
        UIPanel.draw(surface, score_panel, bg_color=(12, 16, 28, 210), border_color=(40, 55, 80))

        score_lbl = self.fonts["small"].render("SCORE", True, (130, 160, 200))
        surface.blit(score_lbl, (35, 28))
        score_val = self.fonts["large"].render(f"{game.score:04d}", True, (0, 235, 255))
        surface.blit(score_val, (35, 48))

        high_lbl = self.fonts["small"].render(f"BEST: {game.high_score}", True, (255, 215, 0))
        surface.blit(high_lbl, (145, 32))

        # Multiplier streak badge
        if game.combo_multiplier > 1:
            streak_txt = f"COMBO x{game.combo_multiplier}"
            streak_surf = self.fonts["small"].render(streak_txt, True, (255, 215, 0))
            surface.blit(streak_surf, (145, 58))

        # 2. Top-Center: Timer & Difficulty
        center_x = self.width // 2
        mins = int(game.game_duration // 60)
        secs = int(game.game_duration % 60)
        time_txt = f"{mins:02d}:{secs:02d}"
        time_surf = self.fonts["medium"].render(time_txt, True, (240, 240, 245))
        time_rect = time_surf.get_rect(center=(center_x, 32))

        timer_panel = (center_x - 90, 16, 180, 48)
        UIPanel.draw(surface, timer_panel, bg_color=(12, 16, 28, 200), border_color=(35, 50, 75))
        surface.blit(time_surf, time_rect)

        diff_badge = f"{game.difficulty.upper()}"
        UIBadge.draw(surface, (center_x - 48, 70), diff_badge, self.fonts["small"], dot_color=(0, 210, 255))

        # 3. Top-Right: Camera PIP (Picture-In-Picture)
        if raw_frame is not None:
            # Annotate frame with hands/landmarks
            annotated = render_hand_annotations(raw_frame, detected_hands, draw_bbox=True, draw_landmarks=True)
            pip_surf = cvframe_to_pygame_surface(annotated, (self.pip_w, self.pip_h))

            # PIP background & border
            pip_border_rect = pygame.Rect(self.pip_x - 2, self.pip_y - 2, self.pip_w + 4, self.pip_h + 4)
            pygame.draw.rect(surface, (0, 200, 240), pip_border_rect, width=2, border_radius=8)
            surface.blit(pip_surf, (self.pip_x, self.pip_y))

            # "LIVE" Tag
            UIBadge.draw(surface, (self.pip_x + 8, self.pip_y + 8), "LIVE CAM",
                         self.fonts["small"], dot_color=(255, 50, 50),
                         bg_color=(10, 10, 15, 220))
        else:
            # Placeholder for disconnected camera
            pip_rect = (self.pip_x, self.pip_y, self.pip_w, self.pip_h)
            UIPanel.draw(surface, pip_rect, bg_color=(15, 20, 30, 220), border_color=(180, 50, 50))
            no_cam = self.fonts["small"].render("NO CAMERA FEED", True, (220, 70, 70))
            surface.blit(no_cam, no_cam.get_rect(center=(self.pip_x + self.pip_w // 2, self.pip_y + self.pip_h // 2)))

        # Lives / Hearts (Right beside or below PIP)
        hearts_x = self.pip_x
        hearts_y = self.pip_y + self.pip_h + 10
        self._draw_hearts(surface, hearts_x, hearts_y, game.lives, game.initial_lives)

        # 4. Bottom-Left: Live Gesture Feedback Card
        card_w, card_h = 340, 76
        card_rect = (20, self.height - card_h - 20, card_w, card_h)
        UIPanel.draw(surface, card_rect, bg_color=(12, 16, 28, 220), border_color=(45, 60, 90))

        # Gesture name and confidence
        dot_color = (0, 255, 150) if gesture.name != "Unknown" else (140, 140, 150)
        pygame.draw.circle(surface, dot_color, (38, self.height - card_h - 20 + 22), 6)

        g_title = f"{gesture.name} ({int(gesture.confidence * 100)}%)"
        g_title_surf = self.fonts["medium"].render(g_title, True, (245, 245, 255))
        surface.blit(g_title_surf, (52, self.height - card_h - 20 + 10))

        # Action description
        act_surf = self.fonts["small"].render(gesture.description, True, (140, 175, 210))
        surface.blit(act_surf, (36, self.height - card_h - 20 + 44))

        # 5. Bottom-Right: System & Engine Metrics
        sys_info = f"Cam: {int(camera_fps)} FPS | {backend_name}"
        UIBadge.draw(surface, (self.width - 240, self.height - 44), sys_info, self.fonts["small"], dot_color=(0, 220, 255))

        # 6. Live Performance Diagnostics Overlay
        self.diagnostics.draw(
            surface=surface,
            camera_fps=camera_fps,
            processing_fps=processing_fps,
            inference_time_ms=inference_time_ms,
            gesture_time_ms=gesture_time_ms,
            latency_ms=latency_ms,
            tracking_status=tracking_status,
            hand_confidence=hand_confidence,
            gesture=gesture,
            cursor_pos=gesture.cursor_pos,
            pos=(20, 115)
        )

        # 7. Pause Overlay Modal
        if game.state == GameState.PAUSED:
            self._draw_pause_modal(surface)

    def _draw_hearts(self, surface: pygame.Surface, x: int, y: int, current_lives: int, max_lives: int) -> None:
        """Draws filled or empty heart symbols for player lives."""
        heart_spacing = 30
        for i in range(max_lives):
            hx = x + i * heart_spacing
            hy = y
            is_alive = i < current_lives
            color = (255, 45, 75) if is_alive else (60, 65, 80)
            # Draw procedural heart using two circles and an inverted triangle
            r = 6
            pygame.draw.circle(surface, color, (hx, hy), r)
            pygame.draw.circle(surface, color, (hx + r * 2, hy), r)
            tri_pts = [(hx - r, hy + 2), (hx + r * 3, hy + 2), (hx + r, hy + r * 2 + 5)]
            pygame.draw.polygon(surface, color, tri_pts)

    def _draw_pause_modal(self, surface: pygame.Surface) -> None:
        """Renders the pause overlay modal dialog."""
        # Semi-transparent dark overlay
        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((5, 8, 16, 200))
        surface.blit(dim_surf, (0, 0))

        # Modal Window Card
        modal_w, modal_h = 420, 320
        modal_rect = (self.width // 2 - modal_w // 2, self.height // 2 - modal_h // 2, modal_w, modal_h)
        UIPanel.draw(surface, modal_rect, bg_color=(15, 20, 36, 245), border_color=(0, 200, 255))

        # Header Text
        title_surf = self.fonts["large"].render("GAME PAUSED", True, (0, 235, 255))
        surface.blit(title_surf, title_surf.get_rect(center=(self.width // 2, self.height // 2 - 105)))

        sub_surf = self.fonts["small"].render("Gesture Tip: Thumbs Up to Resume", True, (160, 185, 220))
        surface.blit(sub_surf, sub_surf.get_rect(center=(self.width // 2, self.height // 2 - 70)))

        # Buttons
        for btn in self.pause_buttons:
            btn.draw(surface)
