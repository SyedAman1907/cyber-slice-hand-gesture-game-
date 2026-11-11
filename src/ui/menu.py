"""
Main Menu Screen Module.
Features cyber-themed layout, animated background particles,
menu action buttons, and live system status indicators.
"""

import math
import random
import time
from typing import Callable, Dict, List, Optional, Tuple
import pygame

from src.ui.components import UIBadge, UIButton, UIPanel


class MenuBackgroundParticle:
    """Subtle floating particle in the background."""
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = random.uniform(0, screen_w)
        self.y = random.uniform(0, screen_h)
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.5, -0.1)
        self.radius = random.uniform(1.5, 3.5)
        self.alpha = random.randint(40, 120)

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        if self.y < -10:
            self.y = self.screen_h + 10
            self.x = random.uniform(0, self.screen_w)
        if self.x < -10:
            self.x = self.screen_w + 10
        elif self.x > self.screen_w + 10:
            self.x = -10

    def draw(self, surface: pygame.Surface) -> None:
        p_surf = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (0, 200, 255, self.alpha), (int(self.radius), int(self.radius)), int(self.radius))
        surface.blit(p_surf, (int(self.x - self.radius), int(self.y - self.radius)))


class MainMenuScreen:
    """Renders and handles the primary navigation menu."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        on_start_game: Callable[[], None],
        on_practice: Callable[[], None],
        on_calibrate: Callable[[], None],
        on_how_to_play: Callable[[], None],
        on_settings: Callable[[], None],
        on_stats: Callable[[], None],
        on_exit: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts

        # Background particles
        self.particles = [MenuBackgroundParticle(screen_w, screen_h) for _ in range(40)]

        # Menu buttons layout
        btn_w, btn_h = 360, 46
        spacing = 11
        start_y = 195
        center_x = screen_w // 2 - btn_w // 2

        self.buttons: List[UIButton] = [
            UIButton(
                (center_x, start_y, btn_w, btn_h),
                "START GAME",
                fonts["medium"],
                bg_color=(20, 80, 120),
                hover_color=(0, 210, 255),
                border_color=(0, 180, 216),
                callback=on_start_game
            ),
            UIButton(
                (center_x, start_y + (btn_h + spacing), btn_w, btn_h),
                "PRACTICE GESTURES",
                fonts["medium"],
                callback=on_practice
            ),
            UIButton(
                (center_x, start_y + (btn_h + spacing) * 2, btn_w, btn_h),
                "CALIBRATE CAMERA & HAND",
                fonts["medium"],
                bg_color=(25, 40, 65),
                hover_color=(0, 200, 240),
                border_color=(0, 180, 220),
                callback=on_calibrate
            ),
            UIButton(
                (center_x, start_y + (btn_h + spacing) * 3, btn_w, btn_h),
                "HOW TO PLAY / GESTURE GUIDE",
                fonts["medium"],
                callback=on_how_to_play
            ),
            UIButton(
                (center_x, start_y + (btn_h + spacing) * 4, btn_w, btn_h),
                "SETTINGS & HARDWARE",
                fonts["medium"],
                callback=on_settings
            ),
            UIButton(
                (center_x, start_y + (btn_h + spacing) * 5, btn_w, btn_h),
                "GAME STATISTICS",
                fonts["medium"],
                callback=on_stats
            ),
            UIButton(
                (center_x, start_y + (btn_h + spacing) * 6, btn_w, btn_h),
                "EXIT",
                fonts["medium"],
                hover_color=(220, 50, 70),
                callback=on_exit
            )
        ]

    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Updates particle motions and button animations."""
        for p in self.particles:
            p.update()

        for btn in self.buttons:
            btn.update(mouse_pos, dt)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Processes mouse clicks and keyboard shortcuts on menu buttons."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self.buttons[0].trigger()
                return

        for btn in self.buttons:
            btn.handle_event(event)

    def trigger_hovered_or_first(self, mouse_pos: Tuple[int, int]) -> bool:
        """Triggers the button currently hovered by cursor (useful for pinch gesture clicks)."""
        for btn in self.buttons:
            if btn.rect.collidepoint(mouse_pos) or btn.is_hovered:
                return btn.trigger()
        return False

    def draw(
        self,
        surface: pygame.Surface,
        camera_status: str = "Connected",
        fps: float = 30.0,
        backend_name: str = "MediaPipe Hands",
        recognized_gesture: str = "None"
    ) -> None:
        """Renders the main menu frame."""
        # Deep charcoal/navy background
        surface.fill((12, 16, 26))

        # Ambient Grid Lines
        grid_color = (20, 26, 42)
        for x in range(0, self.width, 40):
            pygame.draw.line(surface, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 40):
            pygame.draw.line(surface, grid_color, (0, y), (self.width, y), 1)

        # Draw floating particles
        for p in self.particles:
            p.draw(surface)

        # Title Card Panel
        title_y = 60
        title_text = "CYBER SLICE"
        title_surf = self.fonts["title"].render(title_text, True, (0, 235, 255))
        title_rect = title_surf.get_rect(center=(self.width // 2, title_y))

        # Title glow effect
        glow_surf = self.fonts["title"].render(title_text, True, (0, 100, 180))
        surface.blit(glow_surf, (title_rect.x + 2, title_rect.y + 2))
        surface.blit(title_surf, title_rect)

        # Subtitle
        sub_text = "Real-Time Computer Vision & Hand Gesture Gaming"
        sub_surf = self.fonts["small"].render(sub_text, True, (160, 185, 215))
        sub_rect = sub_surf.get_rect(center=(self.width // 2, title_y + 48))
        surface.blit(sub_surf, sub_rect)

        # Central Card Container
        panel_rect = (self.width // 2 - 210, 180, 420, 480)
        UIPanel.draw(surface, panel_rect, bg_color=(16, 22, 38, 200), border_color=(40, 55, 85))

        # Draw Menu Buttons
        for btn in self.buttons:
            btn.draw(surface)

        # Bottom Status Bar
        bar_y = self.height - 44
        pygame.draw.line(surface, (30, 42, 65), (0, bar_y), (self.width, bar_y), 1)

        # Status Badges
        dot_c = (0, 255, 120) if camera_status == "Connected" else (255, 140, 0)
        cam_info = f"Cam: {camera_status} ({int(fps)} FPS)"
        UIBadge.draw(surface, (20, bar_y + 8), cam_info, self.fonts["small"], dot_color=dot_c)

        backend_info = f"Backend: {backend_name}"
        UIBadge.draw(surface, (280, bar_y + 8), backend_info, self.fonts["small"], dot_color=(0, 210, 255))

        gesture_info = f"Live Gesture: {recognized_gesture}"
        UIBadge.draw(surface, (540, bar_y + 8), gesture_info, self.fonts["small"], dot_color=(255, 215, 0))

        # Help hint on the right
        hint_text = "Controls: Move Hand to Slice | Thumbs Up to Pause | Open Palm to Start"
        hint_surf = self.fonts["small"].render(hint_text, True, (130, 150, 180))
        surface.blit(hint_surf, (self.width - hint_surf.get_width() - 20, bar_y + 14))
