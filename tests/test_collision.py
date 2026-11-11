"""
Unit Tests for Continuous Line Segment to Circle Collision Detection.
Validates precision intersection formulas for high-speed blade trails.
"""

import pytest
from src.game.collision import check_trail_fruit_collision, line_segment_circle_intersect
from src.game.fruit import Fruit, FruitType


class TestCollisionDetection:

    def test_direct_line_through_center(self):
        """Segment passing straight through circle center must intersect."""
        p1 = (0.0, 50.0)
        p2 = (100.0, 50.0)
        center = (50.0, 50.0)
        radius = 20.0
        assert line_segment_circle_intersect(p1, p2, center, radius) is True

    def test_tangent_graze(self):
        """Segment touching tangent outer boundary must intersect."""
        p1 = (0.0, 70.0)
        p2 = (100.0, 70.0)
        center = (50.0, 50.0)
        radius = 20.0  # (50 + 20 = 70.0)
        assert line_segment_circle_intersect(p1, p2, center, radius) is True

    def test_segment_miss_above(self):
        """Segment above circle radius must not intersect."""
        p1 = (0.0, 80.0)
        p2 = (100.0, 80.0)
        center = (50.0, 50.0)
        radius = 20.0
        assert line_segment_circle_intersect(p1, p2, center, radius) is False

    def test_segment_endpoint_inside(self):
        """Segment with one endpoint inside circle must intersect."""
        p1 = (55.0, 55.0)  # inside
        p2 = (120.0, 120.0)
        center = (50.0, 50.0)
        radius = 20.0
        assert line_segment_circle_intersect(p1, p2, center, radius) is True

    def test_zero_length_segment(self):
        """Degenerate single-point segment testing."""
        center = (50.0, 50.0)
        radius = 15.0
        # Inside
        assert line_segment_circle_intersect((55.0, 50.0), (55.0, 50.0), center, radius) is True
        # Outside
        assert line_segment_circle_intersect((90.0, 50.0), (90.0, 50.0), center, radius) is False

    def test_trail_fruit_collision_slice(self):
        """Tests continuous trail slicing an active fruit."""
        fruit = Fruit(x=200.0, y=200.0, vx=0.0, vy=0.0, fruit_data=FruitType.WATERMELON)
        trail = [(150.0, 200.0), (250.0, 200.0)]  # Crosses directly through fruit

        hit, vx = check_trail_fruit_collision(trail, fruit, min_slice_speed=2.0, active_slicing=True)
        assert hit is True
        assert vx == 100.0

    def test_trail_slow_speed_ignored(self):
        """Slow drifting hand below speed threshold should not trigger accidental slice."""
        fruit = Fruit(x=200.0, y=200.0, vx=0.0, vy=0.0, fruit_data=FruitType.ORANGE)
        trail = [(199.0, 200.0), (200.0, 200.0)]  # Only 1 pixel movement

        hit, _ = check_trail_fruit_collision(trail, fruit, min_slice_speed=5.0, active_slicing=True)
        assert hit is False