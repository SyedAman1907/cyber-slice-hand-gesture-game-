"""
Game Controller Module.
Decoupled event-driven bridge between recognized hand gestures and game actions.
"""

from typing import Callable, Dict, List, Optional, Tuple
from src.game.game_engine import GameEngine, GameState
from src.gestures.gesture_recognizer import RecognizedGesture
from src.utils.logging_config import setup_logger

logger = setup_logger("GameController")


class GameController:
    """
    Translates recognized gesture events into high-level game actions
    using a configurable action dispatch table.
    """

    def __init__(self, game_engine: GameEngine, action_mapping: Optional[Dict[str, str]] = None):
        self.game = game_engine
        self.action_mapping = action_mapping or {
            "Open Palm": "start_game",
            "Closed Fist": "stop_action",
            "Thumbs Up": "pause_resume",
            "Index Finger Pointing": "cursor_control",
            "Two-Finger Gesture": "secondary_action",
            "Pinch Gesture": "click_interact",
            "Swipe Left": "swipe_left_action",
            "Swipe Right": "swipe_right_action"
        }

        # Event listeners / Callbacks
        self._action_handlers: Dict[str, Callable[[], None]] = {
            "start_game": self._on_start_game,
            "pause_resume": self._on_pause_resume,
            "stop_action": self._on_stop_action,
            "secondary_action": self._on_secondary_action,
            "click_interact": self._on_click_interact,
            "swipe_left_action": self._on_swipe_attack,
            "swipe_right_action": self._on_swipe_attack
        }

        self.last_cursor_pos: Tuple[float, float] = (0.5, 0.5)
        self.is_blade_active = True
        self.last_triggered_action = "none"

    def update_action_mapping(self, new_mapping: Dict[str, str]) -> None:
        """Updates gesture-to-action bindings."""
        self.action_mapping.update(new_mapping)

    def process_gesture(
        self,
        gesture: RecognizedGesture,
        is_tracking_lost: bool = False,
        slicing_mode: str = "motion"
    ) -> None:
        """
        Receives classified gesture from the recognition pipeline
        and applies corresponding actions to the game engine.
        """
        if is_tracking_lost:
            self.game.clear_cursor()
            return

        self.last_cursor_pos = gesture.cursor_pos

        # Resolve whether current gesture satisfies slicing activation
        if slicing_mode == "motion":
            active_slice = True
        elif slicing_mode == "pinch":
            active_slice = (gesture.name == "Pinch Gesture")
        elif slicing_mode == "fist":
            active_slice = (gesture.name == "Closed Fist")
        elif slicing_mode in ("point", "gesture"):
            active_slice = gesture.name in ("Index Finger Pointing", "Two-Finger Gesture")
        else:  # "always"
            active_slice = True

        self.game.update_cursor(gesture.cursor_pos, active_slice=active_slice, slicing_mode=slicing_mode)

        # Trigger action mapped to this gesture
        action_name = self.action_mapping.get(gesture.name, "none")
        if gesture.is_trigger and action_name in self._action_handlers:
            self.last_triggered_action = action_name
            handler = self._action_handlers[action_name]
            handler()

    def _on_start_game(self) -> None:
        """Action handler for start_game."""
        if self.game.state == GameState.GAME_OVER:
            self.game.reset_round()
            logger.info("Controller: Game restarted via gesture.")
        elif self.game.state == GameState.PAUSED:
            self.game.toggle_pause()
            logger.info("Controller: Game unpaused via gesture.")

    def _on_pause_resume(self) -> None:
        """Action handler for pause_resume."""
        if self.game.state in (GameState.PLAYING, GameState.PAUSED):
            self.game.toggle_pause()
            logger.info("Controller: Pause toggled via gesture.")

    def _on_stop_action(self) -> None:
        """Action handler for stop_action (e.g. freeze or shield)."""
        logger.debug("Controller: Stop action triggered.")

    def _on_secondary_action(self) -> None:
        """Action handler for secondary_action (e.g. dual blade or wider slice)."""
        logger.debug("Controller: Secondary blade active.")

    def _on_click_interact(self) -> None:
        """Action handler for click_interact."""
        logger.debug("Controller: Click interaction triggered at cursor.")

    def _on_swipe_attack(self) -> None:
        """Action handler for swipe sweep attack."""
        if self.game.state != GameState.PLAYING:
            return
        logger.info("Controller: Sweep attack triggered! Slicing visible unbombed fruits.")
        # Sweep slice: slice all non-bomb fruits currently in the upper half of screen
        hit_any = False
        for f in self.game.fruits:
            if not f.is_sliced and not f.is_bomb and f.y < self.game.height * 0.7:
                f.slice(hit_vx=10.0)
                self.game.fruits_sliced += 1
                self.game.score += f.fruit_data["points"]
                hit_any = True
        if hit_any:
            self.game.audio.play("combo")
