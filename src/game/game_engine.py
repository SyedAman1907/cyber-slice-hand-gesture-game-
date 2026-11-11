"""
Game Engine Module.
Core Fruit Ninja-style game loop with parabolic fruit waves, combo scoring,
lives, difficulty scaling, procedural sound synthesis, and screen shake.
"""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pygame

from src.game.collision import check_trail_fruit_collision
from src.game.fruit import Fruit, FruitType
from src.utils.logging_config import setup_logger

logger = setup_logger("GameEngine")


class GameState(Enum):
    COUNTDOWN = "Countdown"
    PLAYING = "Playing"
    PAUSED = "Paused"
    GAME_OVER = "Game Over"


@dataclass
class FloatingText:
    """Floating score or combo indicator that drifts up and fades."""
    text: str
    x: float
    y: float
    color: Tuple[int, int, int]
    life: float = 1.0  # 1.0 -> 0.0
    scale: float = 1.0

    def update(self, dt: float) -> bool:
        self.y -= 35.0 * dt
        self.life -= 1.4 * dt
        return self.life > 0.0

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if self.life <= 0:
            return
        alpha = int(max(0, min(255, self.life * 255)))
        txt_surf = font.render(self.text, True, self.color)
        alpha_surf = pygame.Surface(txt_surf.get_size(), pygame.SRCALPHA)
        alpha_surf.blit(txt_surf, (0, 0))
        alpha_surf.set_alpha(alpha)
        rect = alpha_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(alpha_surf, rect)


class SoundSynthesizer:
    """
    Procedural sound generator for retro/arcade audio feedback.
    Synthesizes audio buffers directly using NumPy so the game never
    crashes from missing external .wav or .mp3 audio files.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.enabled = True
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        self._initialized = False

        self._init_mixer()

    def _init_mixer(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=512)
            self._generate_sounds()
            self._initialized = True
            logger.info("Procedural sound synthesizer initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize audio mixer: %s. Continuing in silent mode.", e)
            self.enabled = False

    def _generate_sounds(self) -> None:
        """Synthesizes custom sound effects using waveform formulas."""
        try:
            # Check if mixer initialized as stereo or mono
            channels = 2
            if pygame.mixer.get_init():
                channels = pygame.mixer.get_init()[2]

            def to_channel_format(mono_wave: np.ndarray) -> pygame.mixer.Sound:
                if channels == 2:
                    stereo = np.column_stack((mono_wave, mono_wave))
                    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))
                return pygame.sndarray.make_sound(np.ascontiguousarray(mono_wave))

            # 1. Slice / Whoosh (high-frequency sweep down)
            dur = 0.12
            t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
            freq = np.linspace(1200, 300, len(t))
            env = np.exp(-t * 22)
            wave = (np.sin(2 * np.pi * freq * t) * env * 32767).astype(np.int16)
            self._sounds["slice"] = to_channel_format(wave)

            # 2. Splat / Hit (crunch noise + low pop)
            dur = 0.18
            t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
            noise = np.random.uniform(-1, 1, len(t)) * 0.5
            tone = np.sin(2 * np.pi * 180 * t) * 0.5
            env = np.exp(-t * 18)
            wave = ((noise + tone) * env * 32767).astype(np.int16)
            self._sounds["splat"] = to_channel_format(wave)

            # 3. Bomb Explosion (rumble noise with slow decay)
            dur = 0.45
            t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
            noise = np.random.uniform(-1, 1, len(t))
            env = np.exp(-t * 8)
            wave = (noise * env * 32767).astype(np.int16)
            self._sounds["bomb"] = to_channel_format(wave)

            # 4. Combo Chime (pleasant rising arpeggio tone)
            dur = 0.22
            t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
            tone = np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)
            env = np.exp(-t * 12)
            wave = ((tone / 1.5) * env * 32767).astype(np.int16)
            self._sounds["combo"] = to_channel_format(wave)

            # 5. Game Over (descending sad tone)
            dur = 0.6
            t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
            freq = np.linspace(440, 110, len(t))
            env = np.exp(-t * 4)
            wave = (np.sin(2 * np.pi * freq * t) * env * 32767).astype(np.int16)
            self._sounds["game_over"] = to_channel_format(wave)
        except Exception as e:
            logger.warning("Error generating audio waveforms: %s", e)

    def play(self, sound_name: str) -> None:
        """Plays the designated sound if audio is enabled."""
        if not self.enabled or not self._initialized:
            return
        snd = self._sounds.get(sound_name)
        if snd:
            try:
                snd.play()
            except Exception:
                pass


class GameEngine:
    """
    Core interactive Fruit Ninja game engine with configurable difficulty,
    blade trails, combo multipliers, and persistence integration.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        initial_lives: int = 3,
        difficulty: str = "Medium",
        sound_enabled: bool = True,
        blade_trail_length: int = 16
    ):
        self.width = width
        self.height = height
        self.initial_lives = initial_lives
        self.difficulty = difficulty
        self.blade_trail_length = blade_trail_length

        self.state = GameState.COUNTDOWN
        self.countdown_remaining = 3.0

        # Score & Game Metrics
        self.score = 0
        self.high_score = 0
        self.lives = initial_lives
        self.fruits_sliced = 0
        self.bombs_hit = 0
        self.missed_fruits = 0
        self.game_start_time = time.time()
        self.game_duration = 0.0

        # Combo system
        self.combo_count = 0
        self.combo_timer = 0.0
        self.combo_multiplier = 1

        # Entities & Effects
        self.fruits: List[Fruit] = []
        self.floating_texts: List[FloatingText] = []
        self.cursor_trail: List[Tuple[float, float]] = []

        # Screen shake
        self.shake_time = 0.0
        self.shake_magnitude = 0.0

        # Spawning parameters based on difficulty
        self._setup_difficulty(difficulty)
        self.spawn_timer = 0.0

        # Audio synthesizer
        self.audio = SoundSynthesizer()
        self.audio.enabled = sound_enabled

    def _setup_difficulty(self, diff: str) -> None:
        """Configures gameplay parameters according to selected difficulty."""
        self.difficulty = diff
        if diff == "Easy":
            self.base_spawn_interval = 1.4
            self.bomb_probability = 0.08
            self.max_simultaneous = 2
        elif diff == "Hard":
            self.base_spawn_interval = 0.75
            self.bomb_probability = 0.22
            self.max_simultaneous = 4
        else:  # Medium
            self.base_spawn_interval = 1.05
            self.bomb_probability = 0.15
            self.max_simultaneous = 3

    def reset_round(self) -> None:
        """Resets the gameplay state for a fresh match."""
        self.state = GameState.COUNTDOWN
        self.countdown_remaining = 3.0
        self.score = 0
        self.lives = self.initial_lives
        self.fruits_sliced = 0
        self.bombs_hit = 0
        self.missed_fruits = 0
        self.combo_count = 0
        self.combo_timer = 0.0
        self.combo_multiplier = 1
        self.fruits.clear()
        self.floating_texts.clear()
        self.cursor_trail.clear()
        self.shake_time = 0.0
        self.game_start_time = time.time()
        self.game_duration = 0.0
        self.spawn_timer = 0.0

    def clear_cursor(self) -> None:
        """Immediately wipes cursor trail when hand tracking is lost."""
        self.cursor_trail.clear()

    def update_cursor(
        self,
        normalized_pos: Tuple[float, float],
        active_slice: bool = True,
        slicing_mode: str = "motion"
    ) -> None:
        """
        Updates blade trail using normalized coordinates.
        Applies sub-segment interpolation during rapid movements to prevent missed hits.
        """
        px = normalized_pos[0] * self.width
        py = normalized_pos[1] * self.height

        is_fast_enough = False
        if self.cursor_trail:
            prev_x, prev_y = self.cursor_trail[-1]
            dist = math.hypot(px - prev_x, py - prev_y)
            is_fast_enough = dist >= 3.5

            # Sub-segment interpolation for fast motion to ensure continuous hitboxes
            if dist > 14.0:
                steps = min(12, max(2, int(dist // 10)))
                for s in range(1, steps):
                    t = s / steps
                    ix = prev_x + t * (px - prev_x)
                    iy = prev_y + t * (py - prev_y)
                    self.cursor_trail.append((ix, iy))

            self.cursor_trail.append((px, py))
        else:
            self.cursor_trail.append((px, py))

        while len(self.cursor_trail) > self.blade_trail_length:
            self.cursor_trail.pop(0)

        # Slicing activation resolution
        if slicing_mode == "motion":
            can_slice = is_fast_enough
        elif slicing_mode in ("pinch", "fist", "point", "gesture"):
            can_slice = active_slice
        else:  # "always"
            can_slice = True

        # If currently playing, evaluate collisions along the cursor trail
        if self.state == GameState.PLAYING:
            self._check_slicing_collisions(can_slice)

    def _check_slicing_collisions(self, active_slice: bool) -> None:
        """Checks for collisions between the blade trail and all active fruits."""
        slices_in_frame = 0

        for fruit in self.fruits:
            if fruit.is_sliced:
                continue

            hit, hit_vx = check_trail_fruit_collision(
                self.cursor_trail, fruit, min_slice_speed=3.5, active_slicing=active_slice
            )

            if hit:
                fruit.slice(hit_vx)

                if fruit.is_bomb:
                    self.bombs_hit += 1
                    self.lives -= 1
                    self.shake_screen(0.4, 18.0)
                    self.audio.play("bomb")
                    self.combo_count = 0
                    self.combo_multiplier = 1

                    self.floating_texts.append(
                        FloatingText("BOMB! -1 LIFE", fruit.x, fruit.y, (255, 60, 60), scale=1.4)
                    )

                    if self.lives <= 0:
                        self.trigger_game_over()
                        return
                else:
                    self.fruits_sliced += 1
                    slices_in_frame += 1

                    # Combo progression
                    self.combo_count += 1
                    self.combo_timer = 0.8
                    if self.combo_count >= 5:
                        self.combo_multiplier = 3
                    elif self.combo_count >= 3:
                        self.combo_multiplier = 2
                    else:
                        self.combo_multiplier = 1

                    pts = fruit.fruit_data["points"] * self.combo_multiplier
                    self.score += pts
                    if self.score > self.high_score:
                        self.high_score = self.score

                    self.audio.play("splat")

                    combo_label = f"+{pts}"
                    color = (255, 230, 80) if self.combo_multiplier > 1 else (255, 255, 255)
                    if self.combo_multiplier > 1:
                        combo_label += f" (x{self.combo_multiplier})"
                    self.floating_texts.append(
                        FloatingText(combo_label, fruit.x, fruit.y, color, scale=1.1)
                    )

        if slices_in_frame >= 2:
            self.audio.play("combo")
            self.floating_texts.append(
                FloatingText(f"{slices_in_frame}x MULTI-SLICE!", self.width * 0.5, 180, (0, 255, 200), scale=1.5)
            )

    def trigger_game_over(self) -> None:
        """Transitions state to Game Over."""
        self.state = GameState.GAME_OVER
        self.clear_cursor()
        self.audio.play("game_over")
        logger.info("Game Over! Final Score: %d, Sliced: %d, Bombs: %d",
                    self.score, self.fruits_sliced, self.bombs_hit)

    def toggle_pause(self) -> None:
        """Toggles between PLAYING and PAUSED."""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
            self.clear_cursor()
            logger.info("Game paused.")
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
            self.clear_cursor()
            logger.info("Game resumed.")

    def shake_screen(self, duration: float, magnitude: float) -> None:
        """Triggers a camera shake impulse."""
        self.shake_time = duration
        self.shake_magnitude = magnitude

    def update(self, dt: float) -> None:
        """Advances physics, timers, entity updates, and wave generation."""
        # Update shake
        if self.shake_time > 0:
            self.shake_time -= dt
            if self.shake_time <= 0:
                self.shake_magnitude = 0.0

        # Countdown State
        if self.state == GameState.COUNTDOWN:
            self.countdown_remaining -= dt
            if self.countdown_remaining <= 0:
                self.state = GameState.PLAYING
                self.game_start_time = time.time()
            return

        if self.state != GameState.PLAYING:
            return

        self.game_duration = max(0.0, self.game_duration + dt)

        # Update Combo Timer
        if self.combo_timer > 0:
            self.combo_timer -= dt
            if self.combo_timer <= 0:
                self.combo_count = 0
                self.combo_multiplier = 1

        # Dynamic Difficulty Scaling (Spawn rate speeds up slightly every 100 points)
        score_boost = min(0.40, (self.score // 100) * 0.05)
        current_spawn_interval = max(0.55, self.base_spawn_interval - score_boost)

        # Spawning Waves
        self.spawn_timer += dt
        if self.spawn_timer >= current_spawn_interval:
            self.spawn_timer = 0.0
            num_to_spawn = random.randint(1, self.max_simultaneous)
            for _ in range(num_to_spawn):
                self.fruits.append(
                    Fruit.spawn(self.width, self.height, bomb_probability=self.bomb_probability)
                )

        # Update Entities
        surviving_fruits = []
        for f in self.fruits:
            alive = f.update(dt, self.height)
            if f.missed and not f.is_bomb and not f.is_sliced:
                self.missed_fruits += 1
                # In standard mode, 3 missed fruits = lose 1 life
                if self.missed_fruits % 3 == 0:
                    self.lives -= 1
                    self.floating_texts.append(
                        FloatingText("3 MISSED FRUITS! -1 LIFE", self.width * 0.5, 220, (255, 120, 120))
                    )
                    if self.lives <= 0:
                        self.trigger_game_over()
                        return
            if alive:
                surviving_fruits.append(f)
        self.fruits = surviving_fruits

        # Update Floating texts
        self.floating_texts = [ft for ft in self.floating_texts if ft.update(dt)]

    def render(
        self,
        surface: pygame.Surface,
        font_large: pygame.font.Font,
        font_medium: pygame.font.Font,
        font_small: pygame.font.Font
    ) -> None:
        """Renders all game entities, blade trail, screen shake, and floating text."""
        # Apply Screen Shake Offset
        render_surf = surface
        shake_offset = (0, 0)
        if self.shake_time > 0 and self.shake_magnitude > 0:
            ox = int(random.uniform(-self.shake_magnitude, self.shake_magnitude))
            oy = int(random.uniform(-self.shake_magnitude, self.shake_magnitude))
            shake_offset = (ox, oy)

        # 1. Draw Fruits & Halves
        for fruit in self.fruits:
            fruit.draw(render_surf)

        # 2. Draw Slicing Blade Trail
        self._render_blade_trail(render_surf)

        # 3. Draw Floating Texts
        for ft in self.floating_texts:
            ft.draw(render_surf, font_medium)

        # 4. Draw Countdown Overlay if in COUNTDOWN state
        if self.state == GameState.COUNTDOWN:
            count_num = max(1, int(math.ceil(self.countdown_remaining)))
            txt = str(count_num) if self.countdown_remaining > 0.4 else "GO!"
            color = (255, 215, 0) if count_num > 1 else (0, 255, 150)
            rendered = font_large.render(txt, True, color)
            rect = rendered.get_rect(center=(self.width // 2, self.height // 2))
            # Background glowing circle
            pygame.draw.circle(render_surf, (20, 25, 45, 180), (self.width // 2, self.height // 2), 90)
            render_surf.blit(rendered, rect)

    def _render_blade_trail(self, surface: pygame.Surface) -> None:
        """Renders glowing neon blade trail with tapering thickness."""
        if len(self.cursor_trail) < 2:
            return

        trail_len = len(self.cursor_trail)
        neon_cyan = (0, 235, 255)
        core_white = (255, 255, 255)

        for i in range(trail_len - 1):
            p1 = self.cursor_trail[i]
            p2 = self.cursor_trail[i + 1]

            # Thickness scales up towards cursor tip
            progress = (i + 1) / trail_len
            thickness = max(2, int(progress * 10))

            # Outer neon glow
            pygame.draw.line(surface, neon_cyan, p1, p2, thickness + 4)
            # Inner white blade core
            pygame.draw.line(surface, core_white, p1, p2, max(1, thickness - 2))

        # Tip spark
        tip = self.cursor_trail[-1]
        pygame.draw.circle(surface, (255, 255, 255), (int(tip[0]), int(tip[1])), 6)
        pygame.draw.circle(surface, neon_cyan, (int(tip[0]), int(tip[1])), 9, 2)
