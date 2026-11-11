"""
Game Statistics & Persistence Module.
Handles persistent storage and retrieval of gameplay history and metrics.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger("StatisticsStore")


class StatisticsStore:
    """Manages persistent statistics storage using JSON with safety fallbacks."""

    DEFAULT_DATA: Dict[str, Any] = {
        "summary": {
            "total_games_played": 0,
            "highest_score": 0,
            "total_fruits_sliced": 0,
            "total_bombs_hit": 0,
            "total_play_time_seconds": 0.0,
            "average_score": 0.0
        },
        "recent_games": [],  # List of {id, timestamp, score, fruits_sliced, bombs_hit, duration, difficulty}
        "practice_history": {
            "total_practice_sessions": 0,
            "gestures_tested": {}
        }
    }

    def __init__(self, file_path: str = "data/game_stats.json"):
        self.file_path = Path(file_path)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads data from disk, handling missing or corrupted files gracefully."""
        if not self.file_path.exists():
            logger.info("Stats file '%s' not found. Initializing new store.", self.file_path)
            return self._init_new_file()

        for attempt in range(5):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        logger.warning("Stats file '%s' was empty. Reinitializing.", self.file_path)
                        return self._init_new_file()
                    data = json.loads(content)
                    # Verify required structure
                    if "summary" not in data or "recent_games" not in data:
                        logger.warning("Stats file corrupted schema. Resetting with defaults.")
                        return self._init_new_file()
                    return data
            except (PermissionError, OSError) as e:
                if attempt < 4:
                    time.sleep(0.05)
                    continue
                logger.error("Failed to read stats file (%s). Creating backup and reinitializing.", e)
                try:
                    backup = self.file_path.with_suffix(".corrupt.bak")
                    if self.file_path.exists():
                        os.replace(self.file_path, backup)
                except Exception:
                    pass
                return self._init_new_file()
            except Exception as e:
                logger.error("Failed to read stats file (%s). Creating backup and reinitializing.", e)
                try:
                    backup = self.file_path.with_suffix(".corrupt.bak")
                    if self.file_path.exists():
                        os.replace(self.file_path, backup)
                except Exception:
                    pass
                return self._init_new_file()

        return self._init_new_file()

    def _init_new_file(self) -> Dict[str, Any]:
        """Creates parent directory and writes default empty data."""
        data = json.loads(json.dumps(self.DEFAULT_DATA))
        self._save_to_disk(data)
        return data

    def _save_to_disk(self, data: Dict[str, Any]) -> bool:
        """Atomically saves data to disk with retry handling for Windows file locks."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(f".tmp_{os.getpid()}")
        for attempt in range(5):
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self.file_path)
                return True
            except (PermissionError, OSError):
                time.sleep(0.05)
            except Exception as e:
                logger.error("Error writing stats file: %s", e)
                return False
        # Fallback direct write
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error("Error in fallback stats write: %s", e)
            return False

    def record_game(
        self,
        score: int,
        fruits_sliced: int,
        bombs_hit: int,
        duration_seconds: float,
        difficulty: str = "Medium"
    ) -> Dict[str, Any]:
        """
        Records a completed game round and recalculates all summary statistics.
        
        Returns:
            dict: Summary metrics including whether this was a new high score.
        """
        summary = self.data["summary"]
        is_new_high = score > summary.get("highest_score", 0)

        summary["total_games_played"] += 1
        if is_new_high:
            summary["highest_score"] = score
        summary["total_fruits_sliced"] += fruits_sliced
        summary["total_bombs_hit"] += bombs_hit
        summary["total_play_time_seconds"] = round(summary["total_play_time_seconds"] + duration_seconds, 1)

        # Recalculate average score
        total_games = summary["total_games_played"]
        if total_games > 0:
            current_total = summary.get("_total_score_sum", 0) + score
            summary["_total_score_sum"] = current_total
            summary["average_score"] = round(current_total / total_games, 1)

        game_entry = {
            "id": total_games,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "fruits_sliced": fruits_sliced,
            "bombs_hit": bombs_hit,
            "duration": round(duration_seconds, 1),
            "difficulty": difficulty,
            "is_high_score": is_new_high
        }

        # Keep last 25 games
        recent = self.data.get("recent_games", [])
        recent.insert(0, game_entry)
        self.data["recent_games"] = recent[:25]

        self._save_to_disk(self.data)
        logger.info("Recorded game #%d: Score=%d, HighScore=%s", total_games, score, is_new_high)

        return {
            "score": score,
            "is_new_high": is_new_high,
            "highest_score": summary["highest_score"],
            "total_games": total_games,
            "average_score": summary["average_score"]
        }

    def record_practice_gesture(self, gesture_name: str) -> None:
        """Records practice metrics for a gesture."""
        practice = self.data.setdefault("practice_history", {"total_practice_sessions": 0, "gestures_tested": {}})
        tested = practice.setdefault("gestures_tested", {})
        tested[gesture_name] = tested.get(gesture_name, 0) + 1
        self._save_to_disk(self.data)

    def get_summary(self) -> Dict[str, Any]:
        """Returns lifetime summary statistics."""
        return self.data.get("summary", self.DEFAULT_DATA["summary"].copy())

    def get_recent_games(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent game sessions."""
        return self.data.get("recent_games", [])[:limit]

    def reset_stats(self) -> None:
        """Clears all statistics."""
        self.data = json.loads(json.dumps(self.DEFAULT_DATA))
        self._save_to_disk(self.data)
        logger.info("Statistics have been reset.")
