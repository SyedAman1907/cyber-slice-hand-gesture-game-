"""
UI Components & Graphics Utilities.
Provides reusable dark-theme buttons, glassmorphism panels, badges,
and OpenCV-to-Pygame image conversion.
"""

from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import pygame


def cvframe_to_pygame_surface(cv_frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
    """
    Converts an OpenCV BGR numpy frame into a Pygame Surface.
    
    Args:
        cv_frame: BGR numpy image array.
        target_size: Optional (width, height) to resize to.
        
    Returns:
        pygame.Surface: Displayable surface.
    """
    if cv_frame is None or cv_frame.size == 0:
        surf = pygame.Surface(target_size or (320, 240))
        surf.fill((10, 15, 25))
        return surf

    rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

    if target_size and (w, h) != target_size:
        surf = pygame.transform.smoothscale(surf, target_size)

    return surf


class UIButton:
    """
    Interactive button with hover animations, glow borders,
    and click callback dispatching.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        font: pygame.font.Font,
        bg_color: Tuple[int, int, int] = (25, 32, 50),
        hover_color: Tuple[int, int, int] = (0, 180, 216),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        border_color: Tuple[int, int, int] = (50, 70, 100),
        border_radius: int = 10,
        callback: Optional[Callable[[], None]] = None
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_color = border_color
        self.border_radius = border_radius
        self.callback = callback

        self.is_hovered = False
        self.is_active = False
        self._hover_anim = 0.0  # 0.0 to 1.0 smooth transition

    def update(self, mouse_pos: Tuple[int, int], dt: float = 0.016) -> bool:
        """Updates hover state and animations. Returns True if hovered."""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

        # Smooth animation interpolation
        target = 1.0 if self.is_hovered else 0.0
        self._hover_anim += (target - self._hover_anim) * min(1.0, dt * 15.0)

        return self.is_hovered

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Dispatches click event if button was pressed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) or self.is_hovered:
                self.is_active = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_active and (self.rect.collidepoint(event.pos) or self.is_hovered):
                self.is_active = False
                if self.callback:
                    self.callback()
                return True
            self.is_active = False
        return False

    def trigger(self) -> bool:
        """Programmatically triggers the button callback with visual activation."""
        self.is_active = False
        if self.callback:
            self.callback()
            return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Renders the button with interpolated hover glow."""
        # Interpolate color
        r = int(self.bg_color[0] + (self.hover_color[0] - self.bg_color[0]) * self._hover_anim * 0.4)
        g = int(self.bg_color[1] + (self.hover_color[1] - self.bg_color[1]) * self._hover_anim * 0.4)
        b = int(self.bg_color[2] + (self.hover_color[2] - self.bg_color[2]) * self._hover_anim * 0.4)

        # Draw button surface
        pygame.draw.rect(surface, (r, g, b), self.rect, border_radius=self.border_radius)

        # Glow border on hover
        border_c = (
            int(self.border_color[0] + (0 - self.border_color[0]) * self._hover_anim),
            int(self.border_color[1] + (235 - self.border_color[1]) * self._hover_anim),
            int(self.border_color[2] + (255 - self.border_color[2]) * self._hover_anim)
        ) if self.is_hovered else self.border_color

        border_w = 2 if self.is_hovered else 1
        pygame.draw.rect(surface, border_c, self.rect, width=border_w, border_radius=self.border_radius)

        # Render centered text
        txt_surf = self.font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class UIPanel:
    """Semi-transparent glassmorphic panel container."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        bg_color: Tuple[int, int, int, int] = (15, 20, 35, 220),
        border_color: Tuple[int, int, int] = (45, 60, 90),
        border_radius: int = 12
    ) -> None:
        p_surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(p_surf, bg_color, (0, 0, rect[2], rect[3]), border_radius=border_radius)
        if border_color:
            pygame.draw.rect(p_surf, border_color, (0, 0, rect[2], rect[3]), width=1, border_radius=border_radius)
        surface.blit(p_surf, (rect[0], rect[1]))


class UIBadge:
    """Pill-shaped indicator badge with status dot."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        pos: Tuple[int, int],
        text: str,
        font: pygame.font.Font,
        dot_color: Tuple[int, int, int] = (0, 255, 128),
        bg_color: Tuple[int, int, int, int] = (10, 15, 25, 200),
        border_color: Tuple[int, int, int] = (40, 55, 80)
    ) -> None:
        txt_surf = font.render(text, True, (240, 240, 245))
        w = txt_surf.get_width() + 32
        h = max(26, txt_surf.get_height() + 10)

        badge_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(badge_surf, bg_color, (0, 0, w, h), border_radius=h // 2)
        pygame.draw.rect(badge_surf, border_color, (0, 0, w, h), width=1, border_radius=h // 2)

        # Status dot
        pygame.draw.circle(badge_surf, dot_color, (14, h // 2), 4)

        # Text
        badge_surf.blit(txt_surf, (24, (h - txt_surf.get_height()) // 2))
        surface.blit(badge_surf, pos)
