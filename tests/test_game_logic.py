"""
Unit Tests for Game Mechanics, Fruit Spawning, Scoring, and Game Over Rules.
"""

import pytest
from src.game.fruit import Fruit, FruitType
from src.game.game_engine import GameEngine, GameState


class TestGameLogic:

    def test_fruit_spawning_and_physics(self):
        """Tests fruit creation, velocity, and parabolic gravity updates."""
        fruit = Fruit.spawn(screen_width=1280, screen_height=720, bomb_probability=0.0)
        assert fruit.x > 0 and fruit.x < 1280
        assert fruit.y >= 720  # Spawns from bottom
        assert fruit.vy < 0   # Launched upward
        assert fruit.is_bomb is False

        # Advance physics by 1 second
        prev_y = fruit.y
        fruit.update(dt=0.016, screen_height=720)
        # Position must have advanced
        assert fruit.y != prev_y

    def test_fruit_slicing_and_particles(self):
        """Tests that slicing spawns halves and particle effects."""
        fruit = Fruit(x=300, y=300, vx=2, vy=-5, fruit_data=FruitType.WATERMELON)
        assert fruit.is_sliced is False
        fruit.slice(hit_vx=5.0)
        assert fruit.is_sliced is True
        assert len(fruit.halves) == 2
        assert len(fruit.particles) > 0

    def test_game_engine_scoring_and_combos(self):
        """Tests scoring points, combo multipliers, and high score tracking."""
        engine = GameEngine(width=1280, height=720, sound_enabled=False)
        engine.state = GameState.PLAYING

        # Spawn fruit at known location
        f1 = Fruit(x=400, y=400, vx=0, vy=0, fruit_data=FruitType.APPLE)
        engine.fruits.append(f1)

        # Slice it with cursor trail
        engine.update_cursor((400 / 1280, 390 / 720), active_slice=True)
        engine.update_cursor((400 / 1280, 410 / 720), active_slice=True)

        assert engine.score == FruitType.APPLE["points"]
        assert engine.fruits_sliced == 1
        assert engine.high_score == engine.score

    def test_bomb_hit_penalties(self):
        """Tests that slicing a bomb deducts lives and causes screen shake."""
        engine = GameEngine(width=1280, height=720, initial_lives=3, sound_enabled=False)
        engine.state = GameState.PLAYING

        bomb = Fruit(x=500, y=500, vx=0, vy=0, fruit_data=FruitType.BOMB)
        engine.fruits.append(bomb)

        engine.update_cursor((500 / 1280, 490 / 720), active_slice=True)
        engine.update_cursor((500 / 1280, 510 / 720), active_slice=True)

        assert engine.bombs_hit == 1
        assert engine.lives == 2
        assert engine.shake_time > 0

    def test_game_over_trigger(self):
        """Tests transition to GAME_OVER when lives reach zero."""
        engine = GameEngine(width=1280, height=720, initial_lives=1, sound_enabled=False)
        engine.state = GameState.PLAYING

        bomb = Fruit(x=500, y=500, vx=0, vy=0, fruit_data=FruitType.BOMB)
        engine.fruits.append(bomb)

        engine.update_cursor((500 / 1280, 490 / 720), active_slice=True)
        engine.update_cursor((500 / 1280, 510 / 720), active_slice=True)

        assert engine.lives == 0
        assert engine.state == GameState.GAME_OVER

    def test_pause_and_resume(self):
        """Tests toggling pause state."""
        engine = GameEngine(width=1280, height=720, sound_enabled=False)
        engine.state = GameState.PLAYING
        engine.toggle_pause()
        assert engine.state == GameState.PAUSED
        engine.toggle_pause()
        assert engine.state == GameState.PLAYING
