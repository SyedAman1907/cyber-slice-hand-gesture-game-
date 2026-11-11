"""
Configuration Manager Module.
Loads, validates, and persists application settings.
"""

import json
from pathlib import Path
from typing import Any, Dict
from src.utils.logging_config import setup_logger

logger = setup_logger("ConfigManager")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera": {
        "device_index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "mirrored": True,
        "auto_reconnect": True
    },
    "detection": {
        "backend": "mediapipe",
        "max_num_hands": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.6,
        "yolo_model_path": "models/yolov8n-hand.pt",
        "yolo_confidence_threshold": 0.5,
        "draw_landmarks": True,
        "draw_bounding_box": True
    },
    "gestures": {
        "sensitivity": 1.0,
        "smoothing_factor": 0.5,
        "cooldown_seconds": 0.35,
        "swipe_history_len": 12,
        "swipe_min_distance": 0.18,
        "pinch_threshold": 0.055,
        "finger_curl_threshold": 0.12
    },
    "game": {
        "window_width": 1280,
        "window_height": 720,
        "target_fps": 60,
        "initial_lives": 3,
        "sound_enabled": True,
        "difficulty": "Medium",
        "blade_trail_length": 16,
        "countdown_seconds": 3,
        "action_mapping": {
            "Open Palm": "start_game",
            "Closed Fist": "stop_action",
            "Thumbs Up": "pause_resume",
            "Index Finger Pointing": "cursor_control",
            "Two-Finger Gesture": "secondary_action",
            "Pinch Gesture": "click_interact",
            "Swipe Left": "swipe_left_action",
            "Swipe Right": "swipe_right_action"
        }
    },
    "storage": {
        "stats_file": "data/game_stats.json"
    }
}


class ConfigManager:
    """Manages loading, updating, and persisting configuration settings."""

    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = Path(config_path)
        self.settings: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads configuration from JSON file or creates default if missing/corrupt."""
        if not self.config_path.exists():
            logger.info("Config file '%s' not found. Creating with defaults.", self.config_path)
            self._save_default(DEFAULT_SETTINGS)
            return DEFAULT_SETTINGS.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge with default settings to ensure any newly added keys exist
            merged = self._merge_dicts(DEFAULT_SETTINGS, data)
            return merged
        except Exception as e:
            logger.error("Failed to parse '%s' (%s). Using fallback defaults.", self.config_path, e)
            return DEFAULT_SETTINGS.copy()

    def _save_default(self, data: Dict[str, Any]) -> None:
        """Saves default settings to disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save default config: %s", e)

    def _merge_dicts(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges user dictionary into default dictionary."""
        result = default.copy()
        for k, v in user.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, section: str, key: Any = None, fallback: Any = None) -> Any:
        """Retrieves a configuration value safely."""
        sec = self.settings.get(section)
        if sec is None:
            return key if isinstance(key, dict) else fallback
        if key is None or isinstance(key, dict):
            return sec
        if not isinstance(sec, dict):
            return fallback
        return sec.get(key, fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        """Updates a configuration setting in memory."""
        if section not in self.settings:
            self.settings[section] = {}
        self.settings[section][key] = value

    def save(self) -> bool:
        """Persists current settings to the config JSON file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Configuration saved successfully to %s", self.config_path)
            return True
        except Exception as e:
            logger.error("Error saving configuration: %s", e)
            return False
