"""
How To Play & Gesture Guide Screen Module.
Details the 8 gesture controls, camera positioning tips,
and scoring/survival rules of the game.
"""

from typing import Callable, Dict, Tuple
import pygame

from src.ui.components import UIButton, UIPanel


class HowToPlayScreen:
    """Renders tutorial rules and visual gesture guide."""

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

        cx = screen_w // 2
        self.btn_back = UIButton(
            (cx - 130, screen_h - 75, 260, 48),
            "BACK TO MAIN MENU",
            fonts["medium"],
            bg_color=(20, 80, 120),
            hover_color=(0, 210, 255),
            border_color=(0, 180, 216),
            callback=on_back
        )

    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        self.btn_back.update(mouse_pos, dt)

    def trigger_hovered_button(self, mouse_pos: Tuple[int, int]) -> bool:
        """Triggers hovered button via pinch gesture."""
        if self.btn_back.rect.collidepoint(mouse_pos) or self.btn_back.is_hovered:
            return self.btn_back.trigger()
        return False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.btn_back.trigger()
                return
        self.btn_back.handle_event(event)

    def draw(self, surface: pygame.Surface) -> None:
        """Renders tutorial and gesture guide cards."""
        surface.fill((12, 16, 26))

        cx = self.width // 2

        # 1. Header
        t_surf = self.fonts["large"].render("HOW TO PLAY & GESTURE CONTROLS", True, (0, 235, 255))
        surface.blit(t_surf, t_surf.get_rect(center=(cx, 45)))

        # 2. Left Column: The 8 Gesture Controls Card
        col_w = (self.width - 100) // 2
        card_h = 490
        left_x = 40
        top_y = 85

        UIPanel.draw(surface, (left_x, top_y, col_w, card_h), bg_color=(16, 22, 38, 220), border_color=(45, 65, 95))

        gh_surf = self.fonts["medium"].render("8 HAND GESTURE CONTROLS", True, (255, 215, 0))
        surface.blit(gh_surf, (left_x + 25, top_y + 18))

        gestures_guide = [
            ("1. Open Palm", "All 5 fingers extended", "Start round / Wide blade strike"),
            ("2. Closed Fist", "All fingers curled into palm", "Stop action / Emergency shield"),
            ("3. Thumbs Up", "Thumb upright, 4 fingers curled", "Pause or Resume match"),
            ("4. Index Pointing", "Only index finger extended", "Move cursor & slice fruit"),
            ("5. Two-Finger", "Index & Middle fingers extended", "Dual blade slice"),
            ("6. Pinch", "Touch thumb tip to index fingertip", "Click / Select UI buttons"),
            ("7. Swipe Left", "Fast horizontal sweep to the left", "Trigger left sweep wave"),
            ("8. Swipe Right", "Fast horizontal sweep to the right", "Trigger right sweep wave")
        ]

        row_y = top_y + 54
        for name, pose, effect in gestures_guide:
            # Gesture name
            n_surf = self.fonts["medium"].render(name, True, (0, 235, 255))
            surface.blit(n_surf, (left_x + 25, row_y))

            # How to pose
            p_surf = self.fonts["small"].render(f"Pose: {pose}", True, (215, 225, 240))
            surface.blit(p_surf, (left_x + 195, row_y + 2))

            # Action
            e_surf = self.fonts["small"].render(f"Action: {effect}", True, (130, 160, 195))
            surface.blit(e_surf, (left_x + 195, row_y + 22))

            row_y += 50

        # 3. Right Column: Gameplay Rules & Best Practices Card
        right_x = left_x + col_w + 20
        UIPanel.draw(surface, (right_x, top_y, col_w, card_h), bg_color=(16, 22, 38, 220), border_color=(45, 65, 95))

        rh_surf = self.fonts["medium"].render("GAME RULES & VISION TIPS", True, (255, 215, 0))
        surface.blit(rh_surf, (right_x + 25, top_y + 18))

        tips = [
            ("FRUIT SLICING", "Move your hand so the blade trail intersects flying fruits. Slicing speed must be active!"),
            ("BOMBS AS OBSTACLES", "Avoid slicing black ticking bombs! Hitting a bomb costs 1 life, shakes the screen, and resets combo."),
            ("COMBO MULTIPLIERS", "Slice 3 or more fruits in quick succession to earn 2x and 3x score multipliers."),
            ("MISSED FRUITS", "Do not let fruits fall off screen! Every 3 missed fruits deducts 1 life."),
            ("PLAYER LIVES", "You begin with 3 lives (hearts). When all 3 lives are lost, the match ends."),
            ("CAMERA SETUP TIPS", "Ensure good front lighting. Position your webcam so your full hand is visible within frame. Keep background uncluttered."),
            ("CALIBRATION", "Use 'PRACTICE GESTURES' on the main menu to test all 8 poses before playing.")
        ]

        row_y = top_y + 54
        for title, desc in tips:
            t_surf = self.fonts["medium"].render(f"• {title}", True, (0, 255, 160))
            surface.blit(t_surf, (right_x + 25, row_y))

            # Wrap description into lines
            d_surf = self.fonts["small"].render(desc, True, (190, 205, 225))
            surface.blit(d_surf, (right_x + 40, row_y + 24))

            row_y += 56

        # 4. Back Button
        self.btn_back.draw(surface)
