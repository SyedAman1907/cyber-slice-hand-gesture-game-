"""
Results / Game Over Screen Module.
Displays final score, high score recognition, performance breakdown
(fruits sliced, bombs hit, accuracy %, match duration), and replay navigation.
"""

from typing import Callable, Dict, Tuple
import pygame

from src.ui.components import UIBadge, UIButton, UIPanel


class ResultsScreen:
    """Renders the end-of-round evaluation dashboard."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        on_play_again: Callable[[], None],
        on_menu: Callable[[], None],
        on_stats: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts
        self.on_play_again = on_play_again
        self.on_menu = on_menu
        self.on_stats = on_stats

        btn_w, btn_h = 240, 50
        cx = screen_w // 2
        btn_y = screen_h - 130

        self.btn_play_again = UIButton(
            (cx - btn_w - 140, btn_y, btn_w, btn_h),
            "PLAY AGAIN",
            fonts["medium"],
            bg_color=(20, 80, 120),
            hover_color=(0, 210, 255),
            border_color=(0, 180, 216),
            callback=on_play_again
        )
        self.btn_stats = UIButton(
            (cx - btn_w // 2, btn_y, btn_w, btn_h),
            "VIEW STATS",
            fonts["medium"],
            callback=on_stats
        )
        self.btn_menu = UIButton(
            (cx + 140, btn_y, btn_w, btn_h),
            "MAIN MENU",
            fonts["medium"],
            hover_color=(220, 50, 70),
            callback=on_menu
        )

        self.buttons = [self.btn_play_again, self.btn_stats, self.btn_menu]

    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Updates button hover effects."""
        for btn in self.buttons:
            btn.update(mouse_pos, dt)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Dispatches button clicks and keyboard shortcuts."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_r):
                if self.on_play_again:
                    self.on_play_again()
                return
            elif event.key == pygame.K_ESCAPE:
                if self.on_menu:
                    self.on_menu()
                return
            elif event.key == pygame.K_s:
                if self.on_stats:
                    self.on_stats()
                return

        for btn in self.buttons:
            btn.handle_event(event)

    def trigger_hovered_button(self, mouse_pos: Tuple[int, int]) -> bool:
        """Triggers hovered button via gesture click/pinch."""
        for btn in self.buttons:
            if btn.rect.collidepoint(mouse_pos) or btn.is_hovered:
                return btn.trigger()
        # On results screen, if user pinches anywhere, trigger Play Again
        if self.on_play_again:
            self.on_play_again()
            return True
        return False

    def draw(
        self,
        surface: pygame.Surface,
        score: int,
        fruits_sliced: int,
        bombs_hit: int,
        missed_fruits: int,
        duration_seconds: float,
        difficulty: str,
        is_new_high_score: bool,
        lifetime_high: int
    ) -> None:
        """Renders the results summary."""
        surface.fill((12, 16, 26))

        cx = self.width // 2

        # 1. Title Banner
        title_text = "ROUND COMPLETE" if bombs_hit == 0 else "GAME OVER"
        title_color = (0, 235, 255) if bombs_hit == 0 else (255, 70, 70)
        t_surf = self.fonts["large"].render(title_text, True, title_color)
        surface.blit(t_surf, t_surf.get_rect(center=(cx, 75)))

        # 2. Main Score Container Card
        panel_w, panel_h = 760, 410
        panel_rect = (cx - panel_w // 2, 130, panel_w, panel_h)
        UIPanel.draw(surface, panel_rect, bg_color=(16, 22, 38, 230), border_color=(45, 65, 95))

        # Final Score Header
        lbl_surf = self.fonts["small"].render("FINAL SCORE", True, (140, 170, 210))
        surface.blit(lbl_surf, lbl_surf.get_rect(center=(cx, 165)))

        score_surf = self.fonts["title"].render(f"{score:,}", True, (255, 255, 255))
        surface.blit(score_surf, score_surf.get_rect(center=(cx, 215)))

        # New High Score Badge
        if is_new_high_score:
            UIBadge.draw(surface, (cx - 95, 255), "★ NEW HIGH SCORE! ★", self.fonts["medium"],
                         dot_color=(255, 215, 0), bg_color=(40, 35, 10, 240), border_color=(255, 215, 0))
        else:
            best_txt = f"Personal Best: {lifetime_high:,}"
            best_surf = self.fonts["small"].render(best_txt, True, (255, 215, 0))
            surface.blit(best_surf, best_surf.get_rect(center=(cx, 265)))

        # Separator line
        pygame.draw.line(surface, (35, 48, 72), (cx - panel_w // 2 + 40, 290), (cx + panel_w // 2 - 40, 290), 1)

        # 3. 4-Column Performance Metrics Grid
        total_spawned = fruits_sliced + missed_fruits
        accuracy_pct = int((fruits_sliced / max(1, total_spawned)) * 100)

        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)
        dur_str = f"{mins:02d}:{secs:02d}"

        stats_columns = [
            ("FRUITS SLICED", str(fruits_sliced), (0, 235, 255)),
            ("BOMBS HIT", str(bombs_hit), (255, 75, 75) if bombs_hit > 0 else (120, 220, 120)),
            ("ACCURACY", f"{accuracy_pct}%", (255, 215, 0)),
            ("ROUND TIME", dur_str, (210, 220, 240))
        ]

        col_w = (panel_w - 80) // 4
        start_col_x = cx - panel_w // 2 + 40

        for i, (metric_name, metric_val, val_color) in enumerate(stats_columns):
            box_x = start_col_x + i * col_w
            # Value
            val_surf = self.fonts["large"].render(metric_val, True, val_color)
            surface.blit(val_surf, val_surf.get_rect(center=(box_x + col_w // 2, 345)))
            # Label
            m_surf = self.fonts["small"].render(metric_name, True, (130, 155, 190))
            surface.blit(m_surf, m_surf.get_rect(center=(box_x + col_w // 2, 385)))

        # Difficulty tag inside card
        diff_txt = f"Difficulty Setting: {difficulty.upper()}"
        d_surf = self.fonts["small"].render(diff_txt, True, (110, 140, 175))
        surface.blit(d_surf, d_surf.get_rect(center=(cx, 445)))

        # 4. Action Buttons
        for btn in self.buttons:
            btn.draw(surface)
