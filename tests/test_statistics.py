"""
Unit Tests for Game Statistics Persistence and Error Recovery.
"""

import json
from pathlib import Path
import pytest
from src.data.statistics_store import StatisticsStore


class TestStatisticsStore:

    def test_new_store_initialization(self, tmp_path):
        """Tests that a missing statistics file is initialized with default structure."""
        stats_file = tmp_path / "test_stats.json"
        store = StatisticsStore(str(stats_file))
        assert stats_file.exists()
        summary = store.get_summary()
        assert summary["total_games_played"] == 0
        assert summary["highest_score"] == 0

    def test_record_game_and_metrics_calculation(self, tmp_path):
        """Tests recording games and computing averages/high scores."""
        stats_file = tmp_path / "test_stats.json"
        store = StatisticsStore(str(stats_file))

        # Game 1: 300 pts
        res1 = store.record_game(score=300, fruits_sliced=25, bombs_hit=1, duration_seconds=45.0, difficulty="Medium")
        assert res1["is_new_high"] is True
        assert res1["highest_score"] == 300
        assert res1["total_games"] == 1
        assert res1["average_score"] == 300.0

        # Game 2: 500 pts (New High)
        res2 = store.record_game(score=500, fruits_sliced=40, bombs_hit=0, duration_seconds=60.0, difficulty="Hard")
        assert res2["is_new_high"] is True
        assert res2["highest_score"] == 500
        assert res2["total_games"] == 2
        assert res2["average_score"] == 400.0  # (300 + 500) / 2

        # Game 3: 200 pts (Not a new high)
        res3 = store.record_game(score=200, fruits_sliced=15, bombs_hit=2, duration_seconds=30.0, difficulty="Easy")
        assert res3["is_new_high"] is False
        assert res3["highest_score"] == 500

        # Verify disk persistence by creating new instance on same file
        reloaded = StatisticsStore(str(stats_file))
        summary = reloaded.get_summary()
        assert summary["total_games_played"] == 3
        assert summary["highest_score"] == 500
        assert summary["total_fruits_sliced"] == 80
        assert summary["total_bombs_hit"] == 3

        recent = reloaded.get_recent_games()
        assert len(recent) == 3
        assert recent[0]["score"] == 200  # Most recent first

    def test_corrupt_file_recovery(self, tmp_path):
        """Tests that a corrupted/malformed JSON file resets safely without crash."""
        stats_file = tmp_path / "corrupt_stats.json"
        stats_file.write_text("{This is not valid json content!!!")

        store = StatisticsStore(str(stats_file))
        summary = store.get_summary()
        assert summary["total_games_played"] == 0
        assert stats_file.exists()

    def test_reset_statistics(self, tmp_path):
        """Tests resetting statistics back to empty state."""
        stats_file = tmp_path / "test_stats.json"
        store = StatisticsStore(str(stats_file))
        store.record_game(100, 10, 0, 20.0)
        assert store.get_summary()["total_games_played"] == 1
        store.reset_stats()
        assert store.get_summary()["total_games_played"] == 0
