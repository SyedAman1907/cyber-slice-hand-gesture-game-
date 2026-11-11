"""
Statistics & Analytics Screen Module.
Displays lifetime performance metrics, recent match history,
and gesture practice counts loaded from persistent storage.
"""

from typing import Callable, Dict, Tuple
import pygame

from src.data.statistics_store import StatisticsStore
from src.ui.components import UIButton, UIPanel


class StatisticsScreen:
    """Renders comprehensive game analytics and history dashboard."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fonts: Dict[str, pygame.font.Font],
        stats_store: StatisticsStore,
        on_back: Callable[[], None]
    ):
        self.width = screen_w
        self.height = screen_h
        self.fonts = fonts
        self.store = stats_store
        self.on_back = on_back

        cx = screen_w // 2

        self.btn_back = UIButton(
            (cx - 240, screen_h - 75, 220, 46),
            "BACK TO MENU",
            fonts["medium"],
            callback=on_back
        )
        self.btn_reset = UIButton(
            (cx + 20, screen_h - 75, 220, 46),
            "RESET STATS",
            fonts["medium"],
            hover_color=(220, 60, 60),
            callback=self._reset_stats
        )

        self.buttons = [self.btn_back, self.btn_reset]

    def _reset_stats(self) -> None:
        self.store.reset_stats()

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
            if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.btn_back.trigger()
                return
        for btn in self.buttons:
            btn.handle_event(event)

    def draw(self, surface: pygame.Surface) -> None:
        """Renders the statistics dashboard."""
        surface.fill((12, 16, 26))

        cx = self.width // 2

        # 1. Header
        t_surf = self.fonts["large"].render("GAME STATISTICS & ANALYTICS", True, (0, 235, 255))
        surface.blit(t_surf, t_surf.get_rect(center=(cx, 45)))

        summary = self.store.get_summary()

        # 2. Lifetime Summary KPI Cards (Top Row)
        kpis = [
            ("TOTAL GAMES", f"{summary.get('total_games_played', 0)}", (0, 235, 255)),
            ("HIGHEST SCORE", f"{summary.get('highest_score', 0):,}", (255, 215, 0)),
            ("AVERAGE SCORE", f"{int(summary.get('average_score', 0)):,}", (0, 255, 150)),
            ("FRUITS SLICED", f"{summary.get('total_fruits_sliced', 0):,}", (255, 140, 0)),
            ("BOMBS HIT", f"{summary.get('total_bombs_hit', 0)}", (255, 75, 75)),
            ("PLAY TIME", f"{int(summary.get('total_play_time_seconds', 0) / 60)} min", (180, 200, 240))
        ]

        card_w = (self.width - 80 - 5 * 14) // 6
        card_h = 74
        start_x = 40
        top_y = 80

        for i, (title, val, color) in enumerate(kpis):
            x = start_x + i * (card_w + 14)
            UIPanel.draw(surface, (x, top_y, card_w, card_h), bg_color=(16, 22, 38, 220), border_color=(40, 55, 80))

            v_surf = self.fonts["medium"].render(val, True, color)
            surface.blit(v_surf, v_surf.get_rect(center=(x + card_w // 2, top_y + 26)))

            l_surf = self.fonts["small"].render(title, True, (130, 155, 190))
            surface.blit(l_surf, l_surf.get_rect(center=(x + card_w // 2, top_y + 52)))

        # 3. Recent Matches Table
        table_x = 40
        table_y = 175
        table_w = self.width - 80
        table_h = 420

        UIPanel.draw(surface, (table_x, table_y, table_w, table_h), bg_color=(15, 20, 34, 210), border_color=(40, 55, 80))

        # Table Title
        th_surf = self.fonts["medium"].render("RECENT MATCH SESSIONS", True, (240, 240, 250))
        surface.blit(th_surf, (table_x + 25, table_y + 18))

        recent_games = self.store.get_recent_games(limit=7)

        # Table Column Headers
        col_headers = [
            ("#", 60),
            ("DATE & TIME", 220),
            ("SCORE", 160),
            ("FRUITS SLICED", 160),
            ("BOMBS", 120),
            ("DURATION", 140),
            ("DIFFICULTY", 140)
        ]

        header_y = table_y + 54
        cur_x = table_x + 30
        for h_name, h_w in col_headers:
            h_surf = self.fonts["small"].render(h_name, True, (0, 200, 240))
            surface.blit(h_surf, (cur_x, header_y))
            cur_x += h_w

        pygame.draw.line(surface, (35, 50, 75), (table_x + 20, header_y + 24), (table_x + table_w - 20, header_y + 24), 1)

        # Rows
        row_y = header_y + 32
        if not recent_games:
            no_data = self.fonts["medium"].render("No recorded games yet. Play a round to populate statistics!", True, (120, 140, 170))
            surface.blit(no_data, no_data.get_rect(center=(cx, table_y + 180)))
        else:
            for g in recent_games:
                cur_x = table_x + 30
                is_high = g.get("is_high_score", False)
                row_color = (255, 215, 0) if is_high else (230, 235, 245)

                fields = [
                    str(g.get("id", "-")),
                    g.get("timestamp", "-"),
                    f"{g.get('score', 0):,}",
                    str(g.get("fruits_sliced", 0)),
                    str(g.get("bombs_hit", 0)),
                    f"{int(g.get('duration', 0))}s",
                    str(g.get("difficulty", "Medium"))
                ]

                for val, (_, w) in zip(fields, col_headers):
                    val_surf = self.fonts["small"].render(val, True, row_color)
                    surface.blit(val_surf, (cur_x, row_y))
                    cur_x += w

                row_y += 38

        # 4. Buttons
        for btn in self.buttons:
            btn.draw(surface)
